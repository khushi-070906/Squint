const scanBtn = document.getElementById('scanBtn');
const selfTestBtn = document.getElementById('selfTestBtn');
const statusEl = document.getElementById('status');
const logEl = document.getElementById('log');
const badgeEl = document.getElementById('badge');
const badgeTextEl = document.getElementById('badgeText');
const eyeEl = document.getElementById('eyeSvg');

function closeEye() {
  eyeEl.classList.remove('blink');
  eyeEl.classList.add('scanning');
}

function openEye() {
  // reopen with a quick blink flourish rather than just snapping open
  eyeEl.classList.remove('scanning');
  eyeEl.classList.add('blink');
  setTimeout(() => eyeEl.classList.remove('blink'), 400);
}

scanBtn.addEventListener('click', async () => {
  scanBtn.disabled = true;
  statusEl.textContent = 'Scanning tab…';
  logEl.innerHTML = '';
  closeEye();

  chrome.runtime.sendMessage({ type: 'SQUINT_RUN_SCAN' }, (result) => {
    scanBtn.disabled = false;
    openEye();

    if (!result || result.error) {
      statusEl.textContent = 'Server unreachable — is server/app.py running on :8000?';
      logEl.innerHTML = `<div class="flag">${escapeHtml(result?.error || 'unknown error')}</div>`;
      return;
    }

    statusEl.textContent = `Done · ${result.redactedRegions} region(s) redacted before sending`;
    logEl.innerHTML = renderScanLog(result);
  });
});

selfTestBtn.addEventListener('click', async () => {
  selfTestBtn.disabled = true;
  scanBtn.disabled = true;
  setBadge('idle', 'Running self-test…');
  logEl.innerHTML = '';
  closeEye();

  chrome.runtime.sendMessage({ type: 'SQUINT_SELF_TEST' }, (result) => {
    selfTestBtn.disabled = false;
    scanBtn.disabled = false;
    openEye();

    if (!result) {
      setBadge('fail', 'Self-test did not respond');
      return;
    }

    if (result.pass) {
      setBadge('pass', `Cold Start Self-Test: PASS · ${result.calls.length} call(s), all to localhost`);
    } else {
      setBadge('fail', `Cold Start Self-Test: FAIL · ${result.externalCalls.length} external call(s) detected`);
    }

    const lines = result.calls.map(
      (c) => `<div class="${c.external ? 'flag' : 'ok'}">${c.external ? '✗' : '✓'} ${escapeHtml(c.host)}</div>`
    );
    const combined = lines.join('') || '<div>No network calls made.</div>';
    logEl.innerHTML = combined + (result.scanResult ? renderScanLog(result.scanResult) : '');
  });
});

function renderScanLog(result) {
  const lines = [];
  lines.push(`<div class="ok">✓ ${result.redactedRegions} sensitive region(s) blacked out locally</div>`);
  if (typeof result.faceCount === 'number') {
    lines.push(`<div class="ok">✓ ${result.faceCount} face(s) found by local ONNX detector</div>`);
  }
  if (result.faceDetectorStatus && result.faceDetectorStatus !== 'ok') {
    lines.push(`<div class="flag">face detector: ${escapeHtml(result.faceDetectorStatus)}</div>`);
  }
  if (typeof result.personCount === 'number') {
    lines.push(`<div class="ok">✓ ${result.personCount} person(s) found by local YOLOv8n detector</div>`);
  }
  if (result.personDetectorStatus && result.personDetectorStatus !== 'ok') {
    lines.push(`<div class="flag">person detector: ${escapeHtml(result.personDetectorStatus)}</div>`);
  }
  lines.push(`<div class="ok">✓ sanitized frame sent to server</div>`);
  if (result.server?.summary) {
    lines.push(`<div>server read: "${escapeHtml(result.server.summary)}"</div>`);
  }
  if (result.server?.mode === 'ui_action' && result.server?.action) {
    lines.push(`<div class="flag">action executed: ${escapeHtml(result.server.action.action)} → ${escapeHtml(result.server.action.selector || '')}</div>`);
  }
  if (result.server?.mode === 'processed_data' && result.server?.data) {
    lines.push(`<div>data returned: ${escapeHtml(JSON.stringify(result.server.data))}</div>`);
  }
  if (result.timings) {
    const t = result.timings;
    lines.push(`<div>timings (ms): dom ${t.dom_scan_ms ?? '–'} · capture ${t.capture_ms ?? '–'} · face ${t.face_inference_ms ?? '–'} · person ${t.person_inference_ms ?? '–'} · redact ${t.redaction_ms ?? '–'} · net ${t.network_ms ?? '–'} · total ${t.total_ms ?? '–'}</div>`);
  }
  return lines.join('');
}

function setBadge(state, text) {
  badgeEl.className = `badge ${state}`;
  badgeTextEl.textContent = text;
}

function escapeHtml(str) {
  const d = document.createElement('div');
  d.textContent = str;
  return d.innerHTML;
}
