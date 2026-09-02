Put your local ONNX models here, e.g.:

  models/face_detector.onnx

Run ./download_model.sh from this folder to try fetching one automatically
(best-effort — see the comment block inside the script for manual fallback
links if every source has moved by the time you run it).

A good starting point is a quantized BlazeFace ONNX export (~400KB, runs
comfortably on WebGPU/WASM in-browser). onnx-inference.js expects an
output shaped [N, 5] = (x, y, w, h, score) in normalized [0,1] coordinates,
matching typical BlazeFace export conventions — adjust the postprocessing
in onnx-inference.js if you use a different model/export.

Quick sources to convert/download from:
  - https://github.com/hollance/BlazeFace-PyTorch (convert to ONNX)
  - Any MediaPipe face-detector re-export to ONNX

Also vendor onnxruntime-web here if you don't want a CDN dependency:
  npm i onnxruntime-web
  cp node_modules/onnxruntime-web/dist/*.wasm  extension/models/
  cp node_modules/onnxruntime-web/dist/ort.min.js  extension/
