/**
 * content.js
 * Runs inside the page. Two jobs:
 *   1. Structural PII detection — scan the live DOM for fields that are
 *      sensitive by construction (password inputs, cc-number autocomplete,
 *      email fields, etc). This is cheap and high-precision.
 *   2. Executing action commands that come back from the server
 *      (click / type / scroll), scoped to a whitelist of safe actions.
 */

// ---------- 1. Structural PII detection ----------

const SENSITIVE_SELECTORS = [
  'input[type="password"]',
  'input[autocomplete*="cc-"]',
  'input[autocomplete="email"]',
  'input[type="email"]',
  'input[name*="ssn" i]',
  'input[name*="aadhar" i]',
  'input[name*="card" i]',
  'input[id*="otp" i]',
  '[data-sensitive="true"]'
];

// crude regex-based PII sniffing over visible text nodes (catches PII that
// isn't in a form field — e.g. an email printed as plain text on the page)
const PII_PATTERNS = {
  email: /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g,
  phone: /\b(\+?\d{1,3}[-.\s]?)?\d{10}\b/g,
  card: /\b(?:\d[ -]*?){13,16}\b/g,
  pan: /\b[A-Z]{5}\d{4}[A-Z]\b/g // Indian PAN as an example structured-ID pattern
};

function collectSensitiveRegions() {
  const regions = [];

  // (a) known sensitive form fields
  document.querySelectorAll(SENSITIVE_SELECTORS.join(',')).forEach((el) => {
    const rect = el.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    regions.push({
      type: 'field',
      reason: el.type || el.getAttribute('autocomplete') || 'flagged-selector',
      rect: { x: rect.x, y: rect.y, w: rect.width, h: rect.height }
    });
  });

  // (b) <img>/<video> elements likely containing faces — real face
  // detection happens in onnx-inference.js against a captured frame;
  // here we just flag candidate elements to crop and hand to that model.
  document.querySelectorAll('img, video').forEach((el) => {
    const rect = el.getBoundingClientRect();
    if (rect.width < 24 || rect.height < 24) return; // skip icons
    regions.push({
      type: 'media-candidate',
      reason: 'face-check',
      rect: { x: rect.x, y: rect.y, w: rect.width, h: rect.height }
    });
  });

  // (c) plain-text PII via regex, walking visible text nodes
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let node;
  while ((node = walker.nextNode())) {
    const text = node.nodeValue;
    if (!text || text.trim().length < 4) continue;
    for (const [kind, pattern] of Object.entries(PII_PATTERNS)) {
      pattern.lastIndex = 0;
      if (pattern.test(text)) {
        const parentEl = node.parentElement;
        if (!parentEl) continue;
        const rect = parentEl.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) continue;
        regions.push({
          type: 'text',
          reason: kind,
          rect: { x: rect.x, y: rect.y, w: rect.width, h: rect.height }
        });
      }
    }
  }

  return regions;
}

// ---------- 2. Message handling ----------

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === 'SQUINT_COLLECT_REGIONS') {
    const regions = collectSensitiveRegions();
    sendResponse({
      regions,
      devicePixelRatio: window.devicePixelRatio || 1,
      viewport: { w: window.innerWidth, h: window.innerHeight }
    });
    return true;
  }

  if (msg.type === 'SQUINT_EXECUTE_ACTION') {
    executeAction(msg.action);
    sendResponse({ ok: true });
    return true;
  }
});

// Whitelisted action executor — server can only ask for these primitives,
// never for arbitrary script injection.
function executeAction(action) {
  try {
    switch (action.action) {
      case 'click': {
        const el = document.querySelector(action.selector);
        if (el) el.click();
        break;
      }
      case 'type': {
        const el = document.querySelector(action.selector);
        if (el) {
          el.focus();
          el.value = action.text;
          el.dispatchEvent(new Event('input', { bubbles: true }));
        }
        break;
      }
      case 'scroll': {
        window.scrollBy({ top: action.deltaY || 400, behavior: 'smooth' });
        break;
      }
      case 'wait': {
        // No-op on the page: the server asked for a pause before the next
        // scan (e.g. waiting on a spinner or page transition). background.js
        // handles the actual delay; nothing to execute here.
        break;
      }
      default:
        console.warn('[squint] unknown action ignored:', action.action);
    }
  } catch (e) {
    console.error('[squint] action execution failed', e);
  }
}
