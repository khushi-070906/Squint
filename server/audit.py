"""
audit.py — local SQLite audit trail for Squint/Drishti.

PRD Section 4/5/7: "every redaction, noise injection, and executed action
must be reconstructable from a local log." This module is that log.

Design constraint: the audit DB must be exactly as privacy-safe as the
rest of the pipeline. It stores *metadata about* redactions and actions —
region types, reasons, counts, selectors, VLM summaries — never the raw
image, never field values, never anything that was blacked out. If this
file leaked in full, it should reveal "a password field was redacted and
a click happened on input[type=email]", not any actual password or email.

Storage: a single SQLite file, server/squint_audit.db, created on first
import. No external DB server, no network dependency — matches the
zero-cloud-dependency non-functional requirement.
"""

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Optional

DB_PATH = os.environ.get(
    "SQUINT_AUDIT_DB",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "squint_audit.db"),
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,                  -- unix timestamp, server-side
    task TEXT,                         -- the task string the client sent
    region_count INTEGER NOT NULL,     -- total redacted regions
    region_types TEXT NOT NULL,        -- JSON: [{"type":..,"reason":..,"count":..}]
    face_count INTEGER,
    face_detector_status TEXT,
    noise_epsilon REAL,                -- DP noise budget used
    vlm_mode TEXT,                     -- "ui_action" | "processed_data" | "error"
    vlm_summary TEXT,                  -- model's one-line description of the screen
    vlm_action_json TEXT,              -- JSON action object, if mode == ui_action
    timings_json TEXT,                 -- JSON: dom_scan_ms, capture_ms, face_inference_ms,
                                        --       redaction_ms, network_ms, total_ms
    resources_json TEXT,               -- JSON: client-side resource snapshot (see below)
    error TEXT                         -- non-null if the scan failed
);
CREATE INDEX IF NOT EXISTS idx_scans_ts ON scans(ts);
"""

# columns added after the initial release — kept as an explicit migration
# list (rather than just CREATE TABLE IF NOT EXISTS) so upgrading an
# existing squint_audit.db from an earlier version of this file doesn't
# silently lose the new fields.
_MIGRATIONS = [
    ("resources_json", "TEXT"),
]


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _connect() as conn:
        conn.executescript(_SCHEMA)
        existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(scans)")}
        for col_name, col_type in _MIGRATIONS:
            if col_name not in existing_cols:
                conn.execute(f"ALTER TABLE scans ADD COLUMN {col_name} {col_type}")


def _summarize_region_types(regions: list) -> list:
    """Collapse a raw region list into type/reason counts — never store
    per-region coordinates or values here, just how many of what kind."""
    counts = {}
    for r in regions or []:
        key = (r.get("type", "unknown"), r.get("reason", "unknown"))
        counts[key] = counts.get(key, 0) + 1
    return [
        {"type": t, "reason": reason, "count": c}
        for (t, reason), c in sorted(counts.items())
    ]


def log_scan(
    *,
    task: Optional[str],
    regions: list,
    face_count: Optional[int] = None,
    face_detector_status: Optional[str] = None,
    noise_epsilon: Optional[float] = None,
    vlm_result: Optional[dict] = None,
    timings: Optional[dict] = None,
    resources: Optional[dict] = None,
    error: Optional[str] = None,
) -> int:
    """Insert one audit row for a completed (or failed) /scan call.
    Returns the new row id."""
    region_summary = _summarize_region_types(regions)

    vlm_mode = None
    vlm_summary = None
    vlm_action_json = None
    if vlm_result:
        vlm_mode = vlm_result.get("mode")
        vlm_summary = vlm_result.get("summary")
        if vlm_result.get("action") is not None:
            vlm_action_json = json.dumps(vlm_result["action"])
    if error:
        vlm_mode = vlm_mode or "error"

    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO scans
               (ts, task, region_count, region_types, face_count,
                face_detector_status, noise_epsilon, vlm_mode, vlm_summary,
                vlm_action_json, timings_json, resources_json, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                time.time(),
                task,
                sum(r["count"] for r in region_summary),
                json.dumps(region_summary),
                face_count,
                face_detector_status,
                noise_epsilon,
                vlm_mode,
                vlm_summary,
                vlm_action_json,
                json.dumps(timings) if timings else None,
                json.dumps(resources) if resources else None,
                error,
            ),
        )
        return cur.lastrowid


def get_recent(limit: int = 50) -> list:
    """Most recent scans first, JSON-decoded for easy API/dashboard use."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM scans ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()

    out = []
    for r in rows:
        d = dict(r)
        d["region_types"] = json.loads(d["region_types"]) if d["region_types"] else []
        d["vlm_action"] = json.loads(d["vlm_action_json"]) if d["vlm_action_json"] else None
        d["timings"] = json.loads(d["timings_json"]) if d["timings_json"] else None
        d["resources"] = json.loads(d["resources_json"]) if d.get("resources_json") else None
        del d["vlm_action_json"]
        del d["timings_json"]
        del d["resources_json"]
        out.append(d)
    return out


def get_stats() -> dict:
    """Aggregate counters for a dashboard summary panel."""
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) c FROM scans").fetchone()["c"]
        errors = conn.execute(
            "SELECT COUNT(*) c FROM scans WHERE error IS NOT NULL"
        ).fetchone()["c"]
        avg_total_ms = conn.execute(
            """SELECT AVG(COALESCE(
                   json_extract(timings_json, '$.total_ms'),
                   json_extract(timings_json, '$.approx_total_ms')
               )) a
               FROM scans WHERE timings_json IS NOT NULL"""
        ).fetchone()["a"]
        regions_redacted = conn.execute(
            "SELECT SUM(region_count) s FROM scans"
        ).fetchone()["s"]

    return {
        "total_scans": total,
        "failed_scans": errors,
        "avg_total_latency_ms": round(avg_total_ms, 1) if avg_total_ms else None,
        "total_regions_redacted": regions_redacted or 0,
    }
