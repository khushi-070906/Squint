/**
 * network-monitor.js
 *
 * Powers the "Cold Start Self-Test" badge from PRD Section 6/8: a visible,
 * live proof that zero network calls left the machine except the single
 * POST to the local server.
 *
 * Approach: wrap self.fetch once, in every extension context that could
 * conceivably issue one (background service worker, offscreen document).
 * content.js is intentionally NOT wrapped — and deliberately contains no
 * fetch/XHR calls at all (verifiable by reading it), since it runs inside
 * the page and must never be the thing making network requests.
 *
 * Every call is logged to chrome.storage.local so the popup can read it
 * back after a scan and render a pass/fail badge. This is a code-level
 * guarantee, not a network sniffer: it's honest about what it proves
 * (everything *this extension* deliberately calls) and does not attempt to
 * observe the host page's own traffic, which would require broad
 * <all_urls> host permissions this privacy-focused extension deliberately
 * avoids asking for.
 */

const SQUINT_ALLOWED_HOSTS = ['localhost:8000', '127.0.0.1:8000'];
const SQUINT_LOG_KEY = 'squint_network_log';
const SQUINT_LOG_LIMIT = 50;

(function installMonitor() {
  if (self.__squintFetchWrapped) return;
  self.__squintFetchWrapped = true;

  const nativeFetch = self.fetch.bind(self);

  self.fetch = async function (input, init) {
    const url = typeof input === 'string' ? input : input.url;
    let host = 'unknown';
    let external = true;
    try {
      const u = new URL(url, self.location ? self.location.href : undefined);
      host = u.host;
      // Requests to the extension's own bundled resources (chrome-extension://
      // this-extension-id/...) never touch the network — they're reads of files
      // packaged inside the extension itself (e.g. onnx-inference.js checking
      // whether models/face_detector.onnx exists). Only flag them as external
      // if they somehow point at a *different* extension's origin.
      const isOwnOrigin =
        u.protocol === 'chrome-extension:' &&
        self.location &&
        u.host === new URL(self.location.href).host;
      external = !isOwnOrigin && !SQUINT_ALLOWED_HOSTS.includes(host);
    } catch (e) {
      // data: URLs (used to turn captured screenshots into blobs) aren't
      // "network calls" in any meaningful sense — don't flag them.
      if (typeof url === 'string' && url.startsWith('data:')) {
        external = false;
        host = 'data-url';
      }
    }

    await logCall({ url: safeUrl(url), host, external, ts: Date.now() });
    return nativeFetch(input, init);
  };

  function safeUrl(url) {
    // data: URLs can be megabytes of base64 — never log the payload itself.
    if (typeof url === 'string' && url.startsWith('data:')) return 'data:(inline image)';
    return url;
  }

  async function logCall(entry) {
    try {
      const { [SQUINT_LOG_KEY]: existing = [] } = await chrome.storage.local.get(SQUINT_LOG_KEY);
      const next = [...existing, entry].slice(-SQUINT_LOG_LIMIT);
      await chrome.storage.local.set({ [SQUINT_LOG_KEY]: next });
    } catch (e) {
      console.warn('[squint] network-monitor log failed', e);
    }
  }
})();

async function squintClearNetworkLog() {
  await chrome.storage.local.set({ [SQUINT_LOG_KEY]: [] });
}

async function squintGetNetworkLog() {
  const { [SQUINT_LOG_KEY]: log = [] } = await chrome.storage.local.get(SQUINT_LOG_KEY);
  return log;
}
