/**
 * offscreen.js
 *
 * Runs inside the chrome.offscreen document created by background.js.
 * This is where the PRD's Section 5 gap gets fixed: MV3 service workers
 * cannot touch WebGPU or (reliably) OffscreenCanvas for heavy work, so all
 * of that is delegated here instead. This document is created on demand,
 * does its work, and background.js closes it again — nothing here is
 * user-visible and nothing here makes a network call.
 *
 * Responsibilities:
 *   1. Bitmap redaction (moved from background.js's old inline version).
 *   2. Local ONNX face detection over the captured frame (onnx-inference.js),
 *      wired into the actual pipeline instead of sitting unused in popup.html.
 *   3. Merge any newly-found face boxes into the redaction region list
 *      *before* the bitmap is blacked out, so faces the DOM scan couldn't
 *      see (e.g. a face inside a <canvas> or an inline SVG avatar) still
 *      get redacted.
 */

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type !== 'SQUINT_OFFSCREEN_PROCESS') return false;

  (async () => {
    try {
      const result = await processFrame(msg.dataUrl, msg.regions, msg.dpr || 1);
      sendResponse({ ok: true, ...result });
    } catch (e) {
      sendResponse({ ok: false, error: String(e && e.message ? e.message : e) });
    }
  })();

  return true; // keep the message channel open for the async response
});

async function processFrame(dataUrl, regions, dpr) {
  const timings = {};
  let t0 = performance.now();

  const blob = await (await fetch(dataUrl)).blob();
  const bitmap = await createImageBitmap(blob);
  timings.decode_ms = round(performance.now() - t0);

  // ---- local face detection, best-effort ----
  // If no model file has been placed at extension/models/face_detector.onnx
  // yet, loadModel() rejects and we fall back to DOM-only regions — the
  // pipeline still runs end to end, it just can't catch faces that aren't
  // inside a flagged form field.
  let faceRegions = [];
  let faceDetectorStatus = 'skipped';
  let personRegions = [];
  let personDetectorStatus = 'skipped';
  t0 = performance.now();
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(bitmap, 0, 0);
  const imageData = ctx.getImageData(0, 0, bitmap.width, bitmap.height);

  try {
    const boxes = await self.SquintVision.detectFaces(imageData);
    faceRegions = boxes.map((b) => ({
      type: 'face',
      reason: 'onnx-face-detector',
      rect: { x: b.x / dpr, y: b.y / dpr, w: b.w / dpr, h: b.h / dpr },
      score: b.score
    }));
    faceDetectorStatus = 'ok';
  } catch (e) {
    faceDetectorStatus = 'unavailable: ' + String(e && e.message ? e.message : e);
  }
  timings.face_inference_ms = round(performance.now() - t0);

  // ---- local YOLOv8n person detection, best-effort ----
  // Catches whole-person regions the dedicated face model can miss (back
  // turned, face obscured, sunglasses) — see yolo-inference.js's header
  // for how this was validated. Same graceful-degradation pattern: if
  // extension/models/yolov8n.onnx isn't present, this is skipped and
  // everything else still runs.
  t0 = performance.now();
  try {
    const boxes = await self.SquintYolo.detectPersons(imageData);
    personRegions = boxes.map((b) => ({
      type: 'person',
      reason: 'yolov8n-detector',
      rect: { x: b.x / dpr, y: b.y / dpr, w: b.w / dpr, h: b.h / dpr },
      score: b.score
    }));
    personDetectorStatus = 'ok';
  } catch (e) {
    personDetectorStatus = 'unavailable: ' + String(e && e.message ? e.message : e);
  }
  timings.person_inference_ms = round(performance.now() - t0);

  const allRegions = [...regions, ...faceRegions, ...personRegions];

  // ---- redaction (drawn from EXACT, un-noised coordinates — never jitter
  // these, or a redaction box could shrink/shift and leak a sliver of a
  // real password/card number) ----
  // Reuses the canvas/ctx from the detection step above — it already has
  // the bitmap drawn on it, no need to draw it a second time.
  t0 = performance.now();
  ctx.fillStyle = '#10151C';
  const exactRectsDevicePx = [];
  for (const region of allRegions) {
    const { x, y, w, h } = region.rect;
    const rx = x * dpr, ry = y * dpr, rw = w * dpr, rh = h * dpr;
    ctx.fillRect(rx, ry, rw, rh);
    exactRectsDevicePx.push({ x: rx, y: ry, w: rw, h: rh });
  }
  timings.redaction_ms = round(performance.now() - t0);

  // ---- differential-privacy noise layer (PRD Section 4/6/7) ----
  // Runs strictly AFTER the blackout above, using only the exact rects
  // already painted — see dp-noise.js's header comment for why ordering
  // here is a correctness requirement, not a style choice.
  t0 = performance.now();
  let noiseEpsilon = null;
  try {
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    self.SquintDP.addPixelNoise(imageData, exactRectsDevicePx);
    ctx.putImageData(imageData, 0, 0);
    noiseEpsilon = self.SquintDP.DP_EPSILON_PIXELS;
  } catch (e) {
    // noise is a hardening layer, not a correctness dependency — if it
    // fails for any reason, the frame is still fully redacted, just not
    // additionally noised. Don't fail the whole scan over this.
    console.warn('[squint] pixel noise layer skipped:', e);
  }
  timings.noise_ms = round(performance.now() - t0);

  // metadata copy only — the real allRegions above already did its job
  // drawing the blackout and is discarded; this jittered copy is what
  // actually gets sent to the server.
  const noisyRegions = self.SquintDP.addGeometryNoise(allRegions);

  const redactedBlob = await canvas.convertToBlob({ type: 'image/png' });

  // ---- client resource snapshot (PRD's "client resource utilization"
  // metric) ----
  // background.js is an MV3 service worker with no document context, so
  // performance.memory isn't available there — this offscreen document is
  // the one place in the extension that can actually read it. Best-effort:
  // performance.memory and navigator.deviceMemory are Chrome-only and
  // non-standard, so this degrades gracefully to nulls elsewhere rather
  // than throwing.
  const resources = {
    hardware_concurrency: navigator.hardwareConcurrency ?? null,
    device_memory_gb: navigator.deviceMemory ?? null,
  };
  if (performance.memory) {
    resources.js_heap_used_mb = round(performance.memory.usedJSHeapSize / (1024 * 1024));
    resources.js_heap_limit_mb = round(performance.memory.jsHeapSizeLimit / (1024 * 1024));
  }

  // service workers can't receive Blobs over chrome.runtime messaging in
  // all browsers reliably, so hand it back as a data URL and let
  // background.js turn it into a Blob again for the fetch() call.
  const redactedDataUrl = await blobToDataUrl(redactedBlob);

  return {
    redactedDataUrl,
    allRegions: noisyRegions,
    faceDetectorStatus,
    faceCount: faceRegions.length,
    personDetectorStatus,
    personCount: personRegions.length,
    noiseEpsilon,
    resources,
    timings
  };
}

function blobToDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

function round(n) {
  return Math.round(n * 10) / 10;
}
