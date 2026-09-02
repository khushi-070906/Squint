#!/usr/bin/env bash
# Fetches a small BlazeFace-style ONNX face detector into this folder.
#
# No single source is reliable enough to hardcode blindly (HF links move,
# GitHub LFS objects aren't always fetchable via raw.githubusercontent.com,
# some hosts rate-limit anonymous downloads) — so this tries a short list
# of known sources in order and stops at the first one that produces a
# plausible .onnx file (a real onnx file starts with the bytes "onnx" is
# NOT true — it's protobuf, so we just sanity-check file size instead).
#
# If every source below is stale by the time you run this (models get
# moved/renamed), the manual fallback is:
#   1. https://huggingface.co/garavv/blazeface-onnx  (blaze.onnx, ~400KB,
#      opset 16, input [1,3,128,128], output [N,16] boxes+landmarks + [N] scores)
#   2. https://github.com/PINTO0309/PINTO_model_zoo/tree/main/030_BlazeFace
#      (multiple export variants, browse the folder for a `.onnx` file)
#   3. Any MediaPipe face-detector re-export to ONNX with a similar shape.
#
# Whatever you get, check its actual output tensor shape (onnx model
# tools / netron.app) and adjust the postprocessing loop in
# extension/onnx-inference.js's detectFaces() to match — different
# exports order (x,y,w,h,score) differently or add extra landmark columns.

set -euo pipefail
cd "$(dirname "$0")"

OUT="face_detector.onnx"
MIN_BYTES=50000   # a real quantized BlazeFace export is a few hundred KB;
                   # anything smaller is almost certainly an error page

SOURCES=(
  "https://huggingface.co/garavv/blazeface-onnx/resolve/main/blaze.onnx"
)

for url in "${SOURCES[@]}"; do
  echo "Trying $url ..."
  if curl -fL --max-time 30 -o "$OUT.tmp" "$url" 2>/dev/null; then
    size=$(wc -c < "$OUT.tmp" | tr -d ' ')
    if [ "$size" -ge "$MIN_BYTES" ]; then
      mv "$OUT.tmp" "$OUT"
      echo "Saved $OUT ($size bytes) from $url"
      exit 0
    else
      echo "  got $size bytes — too small, probably not a real model file"
      rm -f "$OUT.tmp"
    fi
  else
    echo "  fetch failed"
  fi
done

echo ""
echo "Could not auto-fetch a model. Grab one manually — see the comment"
echo "block at the top of this script for direct links — and save it as:"
echo "  $(pwd)/$OUT"
echo ""
echo "The extension runs fine without it: DOM-based redaction (passwords,"
echo "emails, cards, OTP fields) still works, it just skips face detection"
echo "until this file exists."
exit 1
