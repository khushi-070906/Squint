"""
Squint server — receives ONLY the already-redacted frame + a sanitized DOM
region list from the extension, forwards it to a VLM (open-weights or
hosted), and returns a structured action for the browser to execute.

This process never sees raw PII: passwords, emails, card numbers and faces
have already been blacked out client-side before the request reaches here.

Run:
    pip install fastapi uvicorn python-multipart pillow httpx
    uvicorn app:app --reload --port 8000
"""

import base64
import json
import os
import time
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import httpx

import audit

app = FastAPI(title="Squint server")

# extension runs from chrome-extension://... origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _init_audit_db():
    audit.init_db()

# ---- VLM backend config ----
# Swap this for a locally hosted open-weights VLM (Qwen2-VL, LLaVA, etc via
# vLLM / ollama / text-generation-inference exposing an OpenAI-compatible
# /v1/chat/completions endpoint), or a hosted API during a live demo.
VLM_API_URL = os.environ.get("VLM_API_URL", "http://localhost:11434/v1/chat/completions")
VLM_MODEL = os.environ.get("VLM_MODEL", "qwen2-vl:7b")
VLM_API_KEY = os.environ.get("VLM_API_KEY", "")  # empty for local models


SYSTEM_PROMPT = """You are a browser automation assistant running on the
user's own machine. You are shown a screenshot where some regions have been
deliberately blacked out by the client BEFORE it ever reached you — these
are intentional PII redactions, not corrupted or missing image data, and
you will never be shown their real contents.

Redaction legend (region "type" / "reason" meanings you will be given
alongside the image — use this to interpret black boxes correctly instead
of guessing or treating them as visual noise):
  - type "field", reason like "password"/"email"/"cc-number"/"otp": a form
    input the client's DOM scan flagged as sensitive by construction.
  - type "text", reason "email"/"phone"/"card"/"pan": plain-text PII the
    client's regex scan found printed on the page (not inside a form field).
  - type "face", reason "onnx-face-detector": a human face the client's
    local vision model located in an image, video, or media element.
  - type "person", reason "yolov8n-detector": a whole person silhouette
    the client's local object detector located — used to catch cases
    (back turned, face obscured, sunglasses) the dedicated face detector
    missed. The redacted box covers the whole person, not just a face.
  - type "media-candidate", reason "face-check": an <img>/<video> element
    that was screened for faces but may not itself contain sensitive content.

Note: region coordinates have a small amount of privacy-preserving noise
applied before reaching you — treat them as approximate (±10px), not
pixel-exact, when reasoning about layout.

Do not guess redacted content and do not mention the redaction itself to
the user. Using the remaining visible layout and the region metadata
(types/reasons/coordinates only, never values), decide the single next
best step to help the user complete their task. Per the official
problem-statement contract, you must return EXACTLY ONE of two response
modes:

  - "ui_action": the next UI action for the client to execute against the
    live page (click, type, scroll, wait). Use this when the task requires
    interacting with the page.
  - "processed_data": a piece of extracted or summarized information for
    the client to re-ingest (e.g. "the total on this invoice is redacted
    but the visible fields say X", or an extracted list/summary). Use this
    when the task is answered by reading the screen, not acting on it.

Respond with strict JSON only, matching this schema:
{
  "summary": "<one sentence describing what you see on screen>",
  "mode": "ui_action" | "processed_data",
  "action": {
    "action": "click" | "type" | "scroll" | "wait" | "none",
    "selector": "<css selector, if click/type>",
    "text": "<text to type, if action is type>",
    "deltaY": <int, if action is scroll>
  },
  "data": "<only present when mode is processed_data; any JSON value>"
}

Omit "action" entirely when mode is "processed_data", and omit "data"
entirely when mode is "ui_action".
"""


@app.post("/scan")
async def scan(
    image: UploadFile = File(...),
    regions: str = Form(...),
    task: Optional[str] = Form(None),
    face_count: Optional[int] = Form(None),
    face_detector_status: Optional[str] = Form(None),
    client_timings: Optional[str] = Form(None),
    noise_epsilon: Optional[float] = Form(None),
    client_resources: Optional[str] = Form(None),
):
    image_bytes = await image.read()

    # sanity-check it's a real, decodable image (defensive — never trust
    # client input blindly even though it's same-origin extension traffic)
    try:
        Image.open(io.BytesIO(image_bytes)).verify()
    except Exception:
        return JSONResponse({"error": "invalid image payload"}, status_code=400)

    region_list = json.loads(regions)
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    timings = json.loads(client_timings) if client_timings else {}
    resources = json.loads(client_resources) if client_resources else None

    user_prompt = (
        f"Task: {task or 'assist the user with the current page'}\n"
        f"Redacted region metadata (types only): {json.dumps(region_list)}\n"
        f"Analyze the attached (partially redacted) screenshot and return the JSON action."
    )

    vlm_start = time.time()
    try:
        result = await call_vlm(image_b64, user_prompt)
    except Exception as e:
        timings["vlm_call_ms"] = round((time.time() - vlm_start) * 1000, 1)
        audit.log_scan(
            task=task,
            regions=region_list,
            face_count=face_count,
            face_detector_status=face_detector_status,
            noise_epsilon=noise_epsilon,
            timings=timings,
            resources=resources,
            error=f"VLM call failed: {e}",
        )
        return JSONResponse({"error": f"VLM call failed: {e}"}, status_code=502)

    timings["vlm_call_ms"] = round((time.time() - vlm_start) * 1000, 1)
    if "total_ms" not in timings:
        # client only has pre-network timings when it builds this request
        # (network/total aren't known until the response comes back), so
        # approximate a total here for the dashboard's latency panel.
        timings["approx_total_ms"] = round(
            sum(v for v in timings.values() if isinstance(v, (int, float))), 1
        )
    audit.log_scan(
        task=task,
        regions=region_list,
        face_count=face_count,
        face_detector_status=face_detector_status,
        noise_epsilon=noise_epsilon,
        timings=timings,
        resources=resources,
        vlm_result=result,
    )

    return JSONResponse(result)


async def call_vlm(image_b64: str, user_prompt: str) -> dict:
    """OpenAI-compatible chat call with an image content block — works
    against vLLM/Ollama/most hosted VLM APIs with minimal changes."""

    headers = {"Content-Type": "application/json"}
    if VLM_API_KEY:
        headers["Authorization"] = f"Bearer {VLM_API_KEY}"

    payload = {
        "model": VLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            },
        ],
        "temperature": 0.2,
        "max_tokens": 400,
    }

    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(VLM_API_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    raw_text = data["choices"][0]["message"]["content"]

    # models sometimes wrap JSON in ```json fences — strip defensively
    cleaned = raw_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # model didn't return valid JSON — degrade to a processed_data
        # response carrying the raw text rather than silently dropping it,
        # so the client still gets *something* usable and auditable.
        parsed = {"summary": raw_text[:200], "mode": "processed_data", "data": raw_text}

    # defensive normalization: never let a malformed response cross the
    # client/server contract, since the client (per PRD Section 6) must be
    # able to trust the dual-mode schema unconditionally.
    if parsed.get("mode") not in ("ui_action", "processed_data"):
        parsed["mode"] = "ui_action" if parsed.get("action") else "processed_data"
    if parsed["mode"] == "ui_action" and not parsed.get("action"):
        parsed["action"] = {"action": "none"}
    if parsed["mode"] == "processed_data" and "data" not in parsed:
        parsed["data"] = None

    return parsed


@app.get("/health")
async def health():
    return {"status": "ok", "vlm_backend": VLM_API_URL}


@app.get("/audit")
async def get_audit(limit: int = 50):
    """Recent scans from the local audit log — region types/counts, face
    detector status, VLM mode/summary/action, and timings. Never includes
    the redacted image or any actual PII value."""
    return JSONResponse({"scans": audit.get_recent(limit=limit)})


@app.get("/audit/stats")
async def get_audit_stats():
    """Aggregate counters for a dashboard summary panel."""
    return JSONResponse(audit.get_stats())
