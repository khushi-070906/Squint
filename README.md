# Squint — Privacy-Preserving Vision Agent (starter prototype)

Local browser extension redacts PII before anything leaves the tab; a
server-side VLM reasons over the sanitized frame and returns an action or
processed data.

```
squint-agent/
├── extension/            Chrome/Edge MV3 extension (client)
│   ├── manifest.json
│   ├── popup.html / popup.js      → UI: scan button + Cold Start Self-Test badge
│   ├── content.js                 → DOM-based PII detection + whitelisted action execution
│   ├── background.js              → orchestrator: capture, delegate to offscreen doc, server call, telemetry
│   ├── offscreen.html / offscreen.js → hosts Canvas redaction + ONNX face detection (needs a real document/GPU context, which the service worker doesn't have)
│   ├── network-monitor.js         → wraps fetch() to power the self-test badge
│   ├── onnx-inference.js          → local ONNX Runtime Web face detector (pluggable model)
│   └── models/                    → drop your .onnx model here; download_model.sh is a best-effort fetcher
└── server/                Python server (server)
    ├── app.py                     → FastAPI, /scan endpoint, redaction-aware VLM prompt, dual-mode response
    └── requirements.txt
```

## What changed from the original prototype

- **Offscreen document.** MV3 service workers can't do GPU/WebGPU work and
  only unreliably support `OffscreenCanvas`. All redaction + face detection
  now runs in `offscreen.html`/`offscreen.js`, created on demand and reused
  across scans. The face detector previously lived in `popup.html` and was
  never actually called — it's now wired into the real scan pipeline.
- **Cold Start Self-Test.** `network-monitor.js` wraps `fetch()` in every
  extension context and logs every call the extension itself makes. The
  popup's "Run Cold Start Self-Test" button clears the log, runs a full
  scan, and shows a pass/fail badge — pass means every network call went to
  `localhost:8000`, nothing left the machine. This is a code-level
  guarantee (only `background.js` ever calls `fetch`; `content.js`
  deliberately contains none), not a packet sniffer — it doesn't need
  broad `<all_urls>` host permissions, which would be an odd ask for a
  privacy-focused extension anyway.
- **Dual-mode server response.** The official SIH26171 brief allows the
  server to return either a UI action *or* processed data for the client to
  re-ingest. `app.py`'s response schema now has an explicit `"mode":
  "ui_action" | "processed_data"` field; `content.js` gained a `wait`
  action to match the brief's named action set.
- **Redaction-aware prompt.** The system prompt now includes an explicit
  legend mapping each region `type`/`reason` to what it means, so the VLM
  interprets black boxes correctly instead of guessing.
- **Latency telemetry.** `background.js` times DOM scan, capture, face
  inference, redaction, and the network round trip, returned in every scan
  result — feeds directly into the Streamlit dashboard work that's still
  to build.

## 1. Run the server

```bash
cd server
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Point at your VLM backend. Any OpenAI-compatible /v1/chat/completions
# endpoint that accepts image_url content works — e.g. Ollama running
# qwen2-vl, vLLM, or a hosted API for a live demo.
export VLM_API_URL=http://localhost:11434/v1/chat/completions
export VLM_MODEL=qwen2-vl:7b
# export VLM_API_KEY=...        # only if your backend needs one

uvicorn app:app --reload --port 8000
```

Sanity check: `curl http://localhost:8000/health`

## 2. Load the extension

1. Go to `chrome://extensions`
2. Enable **Developer mode**
3. **Load unpacked** → select the `extension/` folder
4. Pin the Squint icon, open it on any page with a form

## 3. Add a local face-detection model (optional but part of the spec)

`onnx-inference.js` expects a small ONNX face detector at
`extension/models/face_detector.onnx`. See `extension/models/README.txt`
for where to get/convert one. Without it, the extension still works —
structural DOM-based redaction (passwords, emails, card fields, regex PII)
runs regardless; the ONNX model adds visual face detection on top.

## What actually happens on "Scan & redact"

1. `content.js` scans the live DOM for sensitive fields/text (no network call).
2. `background.js` captures the visible tab as a screenshot.
3. Every flagged region is blacked out **on the bitmap, in the browser**,
   via `OffscreenCanvas` — before any `fetch()` is made.
4. Only the redacted image + region *types* (never values) are POSTed to
   `/scan`.
5. The server prompts a VLM with an explicit note that black regions are
   intentional redactions, and asks for a single next action as JSON.
6. The action (click/type/scroll) is relayed back and executed by
   `content.js` against a whitelist — the server can never inject arbitrary
   script, only pick from these primitives.

## Known gaps to fill in for a full submission

Still open, per the PRD's Section 5/10 roadmap — none of these are done yet:

- **No model file bundled.** Run `extension/models/download_model.sh` or
  see its header comment for manual sources. Without it, `faceDetectorStatus`
  in every scan result will read `"unavailable: ..."` — DOM-based redaction
  still works, but faces embedded in plain `<img>`/`<video>` elements won't
  be caught. `onnx-inference.js`'s postprocessing assumes a `[N,5]`
  `(x,y,w,h,score)` output shape — adjust if your export differs.
- **Client ViT/YOLOv8n full-screen perception** — currently the only visual
  detector is the face model; there's no general "what kind of screen is
  this" classifier yet. This is the highest-weighted judging criterion (25%).
- **Differential-privacy noise layer** — not implemented; the pipeline sends
  the redacted frame as-is, no calibrated noise step yet.
- **Vendored `ort.min.js`** — still CDN-loaded in `offscreen.html`, which
  will hit MV3's default CSP in a packaged (not side-loaded) build. Run
  `npm i onnxruntime-web` and copy the dist files into `extension/` per the
  note in `models/README.txt`.
- **MutationObserver delta scanning, ChromaDB session memory, SQLite audit
  log, LangChain LCEL wrapper, Firefox/WebExtensions polyfill packaging,
  Streamlit dashboard, Docker Compose** — all still to build per the PRD's
  component blueprint (Section 5) and roadmap (Section 10).
- Redaction is still bounding-box blackout only; semantic blurring is a
  one-line change in `offscreen.js`'s `processFrame()` (`ctx.filter =
  'blur(20px)'` + draw instead of `fillRect`) if you want visual continuity.
- `content.js`'s regex PII patterns are a starting set (email/phone/card/PAN)
  — extend for your target PII categories and measure recall/precision
  against a labeled test set for the eval.
