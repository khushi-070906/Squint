/**
 * background.js (MV3 service worker)
 *
 * Orchestrates a scan:
 *   1. Ask content.js for sensitive DOM regions (no network call).
 *   2. Capture the visible tab as an image.
 *   3. Hand the frame + regions to the offscreen document, which runs
 *      local face detection and does the actual bitmap redaction — service
 *      workers can't reliably do WebGPU/Canvas work themselves, which was
 *      the bug in the original prototype (see PRD Section 5/8).
 *   4. POST only the redacted image + a redacted DOM summary (types only,
 *      never values) to the local server.
 *   5. Relay the server's response back to content.js: either a UI action
 *      to execute, or processed data to hand back to the popup.
 *
 * Every stage is timed (performance.now()) for the latency/resource
 * telemetry the PRD calls out as a gap (Section 7 non-functional
 * requirement, 15%-weighted judging criterion).
 */

importScripts('network-monitor.js');

const SERVER_URL = 'http://localhost:8000/scan';
const OFFSCREEN_URL = 'offscreen.html';

let offscreenReady = null;

async function ensureOffscreenDocument() {
  if (offscreenReady) return offscreenReady;

  offscreenReady = (async () => {
    const existing = await chrome.runtime.getContexts?.({
      contextTypes: ['OFFSCREEN_DOCUMENT']
    });
    if (existing && existing.length > 0) return;

    await chrome.offscreen.createDocument({
      url: OFFSCREEN_URL,
      reasons: ['DOM_PARSER', 'BLOBS'],
      justification:
        'Redact captured screenshot pixels and run local ONNX face detection outside the service worker, which has no GPU/Canvas access.'
    });
  })();

  return offscreenReady;
}

async function runScan(tabId, task) {
  const timings = {};
  const overallStart = performance.now();

  // 1. structural regions from the page
  let t0 = performance.now();
  const domReport = await chrome.tabs.sendMessage(tabId, {
    type: 'SQUINT_COLLECT_REGIONS'
  });
  timings.dom_scan_ms = round(performance.now() - t0);

  // 2. capture the visible tab
  t0 = performance.now();
  const dataUrl = await chrome.tabs.captureVisibleTab(null, { format: 'png' });
  timings.capture_ms = round(performance.now() - t0);

  // 3. delegate face detection + redaction to the offscreen document
  await ensureOffscreenDocument();
  t0 = performance.now();
  const offscreenResult = await chrome.runtime.sendMessage({
    type: 'SQUINT_OFFSCREEN_PROCESS',
    dataUrl,
    regions: domReport.regions,
    dpr: domReport.devicePixelRatio
  });
  timings.offscreen_roundtrip_ms = round(performance.now() - t0);

  if (!offscreenResult || !offscreenResult.ok) {
    return {
      error: offscreenResult?.error || 'offscreen processing failed',
      redactedRegions: domReport.regions.length
    };
  }

  Object.assign(timings, offscreenResult.timings);

  // 4. build a sanitized DOM summary — types and reasons only, no values —
  // plus a short legend so the server knows what each redaction means
  // instead of treating black boxes as corrupted image data.
  const sanitizedDom = offscreenResult.allRegions.map((r) => ({
    type: r.type,
    reason: r.reason,
    rect: r.rect // layout is not sensitive, only content is
  }));

  const formData = new FormData();
  const redactedBlob = await (await fetch(offscreenResult.redactedDataUrl)).blob();
  formData.append('image', redactedBlob, 'frame.png');
  formData.append('regions', JSON.stringify(sanitizedDom));
  formData.append('task', task || 'assist-user');
  // Metadata only (never redacted values) — lets the server's audit log
  // record the full picture: how many faces, how the pipeline timed out,
  // not just the sanitized frame itself.
  formData.append('face_count', String(offscreenResult.faceCount ?? 0));
  formData.append('face_detector_status', offscreenResult.faceDetectorStatus || 'unknown');
  formData.append('client_timings', JSON.stringify(timings));
  if (offscreenResult.noiseEpsilon != null) {
    formData.append('noise_epsilon', String(offscreenResult.noiseEpsilon));
  }
  if (offscreenResult.resources) {
    formData.append('client_resources', JSON.stringify(offscreenResult.resources));
  }

  let serverResponse;
  t0 = performance.now();
  try {
    const res = await fetch(SERVER_URL, { method: 'POST', body: formData });
    serverResponse = await res.json();
  } catch (e) {
    console.error('[squint] server call failed', e);
    return { error: String(e), redactedRegions: sanitizedDom.length, timings };
  }
  timings.network_ms = round(performance.now() - t0);
  timings.total_ms = round(performance.now() - overallStart);

  // 5. dual-mode response: either a UI action to execute, or processed
  // data to hand back to the popup for the user/next step to consume.
  if (serverResponse.mode === 'ui_action' && serverResponse.action) {
    await chrome.tabs.sendMessage(tabId, {
      type: 'SQUINT_EXECUTE_ACTION',
      action: serverResponse.action
    });
  }

  return {
    redactedRegions: sanitizedDom.length,
    faceCount: offscreenResult.faceCount,
    faceDetectorStatus: offscreenResult.faceDetectorStatus,
    personCount: offscreenResult.personCount,
    personDetectorStatus: offscreenResult.personDetectorStatus,
    server: serverResponse,
    timings
  };
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === 'SQUINT_RUN_SCAN') {
    chrome.tabs.query({ active: true, currentWindow: true }, async ([tab]) => {
      try {
        const result = await runScan(tab.id, msg.task);
        sendResponse(result);
      } catch (e) {
        console.error('[squint] scan failed', e);
        sendResponse({
          error: 'Scan failed: ' + String(e && e.message ? e.message : e) +
            ' — if you just (re)loaded the extension, refresh this tab and try again.'
        });
      }
    });
    return true; // keep the message channel open for the async response
  }

  if (msg.type === 'SQUINT_SELF_TEST') {
    (async () => {
      try {
        await squintClearNetworkLog();
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        const scanResult = await runScan(tab.id, 'self-test');
        const log = await squintGetNetworkLog();
        const external = log.filter((e) => e.external);
        sendResponse({
          pass: external.length === 0,
          calls: log,
          externalCalls: external,
          scanResult
        });
      } catch (e) {
        console.error('[squint] self-test failed', e);
        sendResponse({
          pass: false,
          calls: [],
          externalCalls: [],
          error: 'Self-test failed: ' + String(e && e.message ? e.message : e) +
            ' — if you just (re)loaded the extension, refresh this tab and try again.'
        });
      }
    })();
    return true;
  }
});

function round(n) {
  return Math.round(n * 10) / 10;
}
