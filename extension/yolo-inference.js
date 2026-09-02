/**
 * yolo-inference.js
 *
 * Client-side general object detection (PRD Section 4/5: "Client ViT +
 * YOLOv8n perception") — catches redaction-worthy content that neither
 * DOM scanning nor the dedicated face detector can: a person's silhouette
 * from behind, a face turned away or wearing sunglasses, a person visible
 * in a video-call thumbnail or embedded photo where onnx-inference.js's
 * face model didn't fire.
 *
 * Model: a standard Ultralytics YOLOv8n export (COCO, 80 classes),
 * input "images" [1,3,640,640] float32, output "output0" [1,84,8400]
 * (4 box params + 80 class scores per anchor, no NMS baked in — NMS runs
 * here in JS). This exact model + the pre/post-processing below was
 * verified against a real photo (Ultralytics' bus.jpg test image) before
 * being ported here: it correctly found and tightly boxed all 4 people
 * in the image after NMS. Only the "person" class is used for redaction
 * — other COCO classes (car, dog, etc.) aren't privacy-relevant here.
 *
 * Shares the same vendored ort.min.js and the same graceful-degradation
 * pattern as onnx-inference.js: if extension/models/yolov8n.onnx isn't
 * present, this fails fast with a clear message and the rest of the
 * pipeline (DOM redaction + face detector) still runs unaffected.
 */

const YOLO_MODEL_URL = chrome.runtime.getURL('models/yolov8n.onnx');
const YOLO_INPUT_SIZE = 640;
const PERSON_CLASS_ID = 0; // COCO class 0 == "person"
const LETTERBOX_PAD_COLOR = 114; // matches Ultralytics' own preprocessing convention

let yoloSession = null;
let yoloLoadFailed = null;

async function loadYoloModel() {
  if (yoloSession) return yoloSession;
  if (yoloLoadFailed) throw yoloLoadFailed;

  try {
    const head = await fetch(YOLO_MODEL_URL, { method: 'HEAD' });
    if (!head.ok) throw new Error('not found');
  } catch (e) {
    yoloLoadFailed = new Error(
      'No yolov8n.onnx found at extension/models/. Person-detection redaction ' +
      'is skipped — DOM + face-detector redaction still runs without it.'
    );
    throw yoloLoadFailed;
  }

  const ort = self.ort;
  if (!ort) {
    yoloLoadFailed = new Error('ort.min.js did not load (check offscreen.html script tags)');
    throw yoloLoadFailed;
  }

  try {
    yoloSession = await ort.InferenceSession.create(YOLO_MODEL_URL, {
      executionProviders: ['webgpu', 'wasm']
    });
  } catch (e) {
    console.warn('[squint] YOLO WebGPU unavailable, falling back to wasm', e);
    yoloSession = await ort.InferenceSession.create(YOLO_MODEL_URL, {
      executionProviders: ['wasm']
    });
  }
  return yoloSession;
}

/**
 * Letterbox-resize ImageData into a YOLO_INPUT_SIZE square (aspect ratio
 * preserved, padded with gray) and produce the NCHW float32 tensor the
 * model expects. Returns the tensor plus the scale/pad values needed to
 * map detected boxes back to the original image's pixel coordinates.
 */
function preprocess(imageData) {
  const { width: srcW, height: srcH } = imageData;

  const srcCanvas = new OffscreenCanvas(srcW, srcH);
  srcCanvas.getContext('2d').putImageData(imageData, 0, 0);

  const scale = Math.min(YOLO_INPUT_SIZE / srcW, YOLO_INPUT_SIZE / srcH);
  const newW = Math.round(srcW * scale);
  const newH = Math.round(srcH * scale);
  const padX = Math.floor((YOLO_INPUT_SIZE - newW) / 2);
  const padY = Math.floor((YOLO_INPUT_SIZE - newH) / 2);

  const dstCanvas = new OffscreenCanvas(YOLO_INPUT_SIZE, YOLO_INPUT_SIZE);
  const dstCtx = dstCanvas.getContext('2d');
  dstCtx.fillStyle = `rgb(${LETTERBOX_PAD_COLOR},${LETTERBOX_PAD_COLOR},${LETTERBOX_PAD_COLOR})`;
  dstCtx.fillRect(0, 0, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE);
  dstCtx.drawImage(srcCanvas, 0, 0, srcW, srcH, padX, padY, newW, newH);

  const letterboxed = dstCtx.getImageData(0, 0, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE);
  const { data } = letterboxed;
  const size = YOLO_INPUT_SIZE;
  const float32Data = new Float32Array(3 * size * size);

  // HWC RGBA -> CHW RGB, normalized to [0, 1]
  for (let i = 0; i < size * size; i++) {
    const srcIdx = i * 4;
    float32Data[i] = data[srcIdx] / 255;                       // R plane
    float32Data[size * size + i] = data[srcIdx + 1] / 255;     // G plane
    float32Data[2 * size * size + i] = data[srcIdx + 2] / 255; // B plane
  }

  const tensor = new self.ort.Tensor('float32', float32Data, [1, 3, size, size]);
  return { tensor, scale, padX, padY };
}

/** Greedy NMS — same algorithm validated in the Python reference. */
function nms(boxes, scores, iouThreshold) {
  const order = scores
    .map((s, i) => i)
    .sort((a, b) => scores[b] - scores[a]);
  const areas = boxes.map((b) => (b.x2 - b.x1) * (b.y2 - b.y1));
  const keep = [];
  const suppressed = new Set();

  for (let idx = 0; idx < order.length; idx++) {
    const i = order[idx];
    if (suppressed.has(i)) continue;
    keep.push(i);
    for (let jdx = idx + 1; jdx < order.length; jdx++) {
      const j = order[jdx];
      if (suppressed.has(j)) continue;
      const xx1 = Math.max(boxes[i].x1, boxes[j].x1);
      const yy1 = Math.max(boxes[i].y1, boxes[j].y1);
      const xx2 = Math.min(boxes[i].x2, boxes[j].x2);
      const yy2 = Math.min(boxes[i].y2, boxes[j].y2);
      const w = Math.max(0, xx2 - xx1);
      const h = Math.max(0, yy2 - yy1);
      const inter = w * h;
      const iou = inter / (areas[i] + areas[j] - inter);
      if (iou > iouThreshold) suppressed.add(j);
    }
  }
  return keep;
}

/**
 * Run person detection over an ImageData captured from the tab (same
 * ImageData object offscreen.js already builds for the face detector —
 * no extra canvas draw needed by the caller). Returns boxes in the
 * *same pixel space as imageData* (i.e. bitmap/device pixels) — offscreen.js
 * divides by dpr when merging these into the region list, exactly as it
 * already does for face boxes.
 */
async function detectPersons(imageData, { confThreshold = 0.45, iouThreshold = 0.45 } = {}) {
  const session = await loadYoloModel();
  const { tensor, scale, padX, padY } = preprocess(imageData);

  const results = await session.run({ images: tensor });
  const output = results[Object.keys(results)[0]]; // [1, 84, 8400]
  const [, numAttrs, numAnchors] = output.dims;
  const data = output.data;

  // output is laid out attribute-major: data[attr*numAnchors + anchor].
  // Walk anchors, and for each, pull out its 4 box params + 80 class
  // scores by striding through that layout (equivalent to the Python
  // reference's `preds = out[0].T` transpose, done here without
  // materializing a transposed copy).
  const numClasses = numAttrs - 4;
  const candidateBoxes = [];
  const candidateScores = [];

  for (let a = 0; a < numAnchors; a++) {
    // class scores start at attribute index 4; find the best one
    let bestScore = -Infinity;
    let bestClass = -1;
    for (let c = 0; c < numClasses; c++) {
      const score = data[(4 + c) * numAnchors + a];
      if (score > bestScore) {
        bestScore = score;
        bestClass = c;
      }
    }
    if (bestClass !== PERSON_CLASS_ID || bestScore < confThreshold) continue;

    const cx = data[0 * numAnchors + a];
    const cy = data[1 * numAnchors + a];
    const bw = data[2 * numAnchors + a];
    const bh = data[3 * numAnchors + a];

    candidateBoxes.push({
      x1: cx - bw / 2,
      y1: cy - bh / 2,
      x2: cx + bw / 2,
      y2: cy + bh / 2
    });
    candidateScores.push(bestScore);
  }

  const keepIdx = nms(candidateBoxes, candidateScores, iouThreshold);

  // un-letterbox: map from the 640x640 padded/scaled space back to the
  // original imageData's pixel coordinates.
  return keepIdx.map((i) => {
    const b = candidateBoxes[i];
    const x1 = (b.x1 - padX) / scale;
    const y1 = (b.y1 - padY) / scale;
    const x2 = (b.x2 - padX) / scale;
    const y2 = (b.y2 - padY) / scale;
    return {
      x: Math.max(0, x1),
      y: Math.max(0, y1),
      w: Math.max(1, x2 - x1),
      h: Math.max(1, y2 - y1),
      score: candidateScores[i]
    };
  });
}

self.SquintYolo = { loadYoloModel, detectPersons };
