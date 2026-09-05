/**
 * onnx-inference.js
 *
 * Local, in-browser vision model. Two things happen here:
 *   1. A lightweight face detector (BlazeFace-style, ONNX) runs over the
 *      captured screenshot to find face bounding boxes — this catches PII
 *      that structural DOM scanning misses (e.g. a face inside a plain <img>
 *      or a video call thumbnail).
 *   2. (Optional, heavier) A small ViT can classify screen regions to help
 *      the "visual context" score — e.g. "this looks like a payment form" —
 *      which content.js's selector-based detection can't infer on its own.
 *
 * Uses ONNX Runtime Web with the WebGPU execution provider, falling back to
 * WASM automatically if WebGPU isn't available. Swap MODEL_URL for a real
 * quantized model (e.g. a BlazeFace .onnx export) before shipping.
 *
 * npm i onnxruntime-web   (bundle it, or load from a CDN as done below)
 */

// Shared onnxruntime-web config, applied once here (this file loads before
// yolo-inference.js in offscreen.html) so it's in effect for every model
// this extension runs. numThreads=1 forces the single-threaded WASM path —
// the multi-threaded build needs SharedArrayBuffer, which needs
// cross-origin-isolation headers (COOP/COEP) a chrome-extension:// page
// doesn't have configured. Without this, session creation fails with
// "no available backend found" / "previous call to initWasm() failed" —
// reproduced and fixed against the real yolov8n.onnx model in a headless
// Chromium test before shipping this.
if (self.ort && self.ort.env && self.ort.env.wasm) {
  self.ort.env.wasm.numThreads = 1;
}

let ortSession = null;
let modelLoadFailed = null; // cache a definitive failure so we don't retry every scan
const MODEL_URL = chrome.runtime.getURL('models/face_detector.onnx');

async function loadModel() {
  if (ortSession) return ortSession;
  if (modelLoadFailed) throw modelLoadFailed;

  // Fail fast and with a clear message if no model has been placed yet,
  // rather than letting ort.InferenceSession.create() throw an opaque
  // parse error. See extension/models/README.txt / download_model.sh.
  try {
    const head = await fetch(MODEL_URL, { method: 'HEAD' });
    if (!head.ok) throw new Error('not found');
  } catch (e) {
    modelLoadFailed = new Error(
      'No face_detector.onnx found at extension/models/. Run models/download_model.sh ' +
      'or drop a BlazeFace-style ONNX export there — DOM-based redaction still runs without it.'
    );
    throw modelLoadFailed;
  }

  // Loaded lazily from a CDN build of onnxruntime-web in this scaffold;
  // for a production extension, vendor ort.min.js + ort-wasm files locally
  // since Manifest V3 restricts remote code execution. Runs inside the
  // offscreen document (offscreen.html), not the popup, so it survives
  // past the popup closing.
  const ort = self.ort;
  if (!ort) {
    modelLoadFailed = new Error('ort.min.js did not load (check offscreen.html script tags)');
    throw modelLoadFailed;
  }

  try {
    ortSession = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ['webgpu', 'wasm']
    });
    console.log('[squint] model loaded on', ortSession.handler?.provider ?? 'unknown backend');
  } catch (e) {
    console.warn('[squint] WebGPU unavailable, falling back to wasm', e);
    ortSession = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ['wasm']
    });
  }
  return ortSession;
}

/**
 * Run face detection over an ImageBitmap/ImageData captured from the tab.
 * Returns an array of { x, y, w, h, score } in image pixel coordinates.
 */
async function detectFaces(imageData, { inputSize = 128, scoreThreshold = 0.6 } = {}) {
  const session = await loadModel();

  const tensor = imageDataToTensor(imageData, inputSize);
  const feeds = { input: tensor };
  const results = await session.run(feeds);

  // Shape of outputs depends on the exported model; this assumes a
  // typical [N, 5] output of (x, y, w, h, score) in normalized coords.
  const output = results[Object.keys(results)[0]];
  const boxes = [];
  const data = output.data;
  const stride = 5;
  for (let i = 0; i < data.length; i += stride) {
    const score = data[i + 4];
    if (score < scoreThreshold) continue;
    boxes.push({
      x: data[i] * imageData.width,
      y: data[i + 1] * imageData.height,
      w: data[i + 2] * imageData.width,
      h: data[i + 3] * imageData.height,
      score
    });
  }
  return boxes;
}

function imageDataToTensor(imageData, size) {
  // Resize + normalize to [1,3,size,size] Float32 — implement with an
  // OffscreenCanvas resize pass in production. Left minimal here since the
  // exact preprocessing depends on the chosen model's training pipeline.
  const { data, width, height } = imageData;
  const float32Data = new Float32Array(3 * size * size);

  const scaleX = width / size;
  const scaleY = height / size;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const srcX = Math.floor(x * scaleX);
      const srcY = Math.floor(y * scaleY);
      const srcIdx = (srcY * width + srcX) * 4;
      const dstIdx = y * size + x;

      float32Data[dstIdx] = (data[srcIdx] / 255 - 0.5) / 0.5;                     // R
      float32Data[size * size + dstIdx] = (data[srcIdx + 1] / 255 - 0.5) / 0.5;   // G
      float32Data[2 * size * size + dstIdx] = (data[srcIdx + 2] / 255 - 0.5) / 0.5; // B
    }
  }

  return new ort.Tensor('float32', float32Data, [1, 3, size, size]);
}

self.SquintVision = { loadModel, detectFaces };
