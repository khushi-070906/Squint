"""
dashboard.py — Streamlit telemetry dashboard for Squint/Drishti.

PRD Section 12.1 judging weights this covers:
  - 20% "client resource utilization"  -> Resource panel
  - 20% "redaction precision"          -> Redaction panel (counts/types;
                                           true tight-vs-loose IoU needs a
                                           labeled eval set, flagged below)
  - 15% "end-to-end latency"           -> Latency panel
  - plus a general audit/activity view for the "every action reconstructable
    from a local log" requirement.

Reads directly from server/squint_audit.db (read-only) — no dependency on
the FastAPI server being up, though it usually will be since that's what's
writing to the DB. Zero external network calls, matches the rest of the
project's zero-cloud-dependency stance.

Run:
    pip install streamlit pandas
    streamlit run dashboard.py
"""

import json
import os
import sqlite3

import pandas as pd
import streamlit as st

DB_PATH = os.environ.get(
    "SQUINT_AUDIT_DB",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "squint_audit.db"),
)

st.set_page_config(page_title="Squint — Telemetry", layout="wide")


@st.cache_data(ttl=5)
def load_scans() -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM scans ORDER BY ts DESC").fetchall()
    conn.close()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([dict(r) for r in rows])
    df["timestamp"] = pd.to_datetime(df["ts"], unit="s")

    def safe_json(x):
        try:
            return json.loads(x) if x else {}
        except (TypeError, json.JSONDecodeError):
            return {}

    df["region_types_parsed"] = df["region_types"].apply(safe_json)
    df["timings_parsed"] = df["timings_json"].apply(safe_json)
    df["resources_parsed"] = df["resources_json"].apply(safe_json)

    df["total_ms"] = df["timings_parsed"].apply(
        lambda t: t.get("total_ms") or t.get("approx_total_ms")
    )
    df["js_heap_used_mb"] = df["resources_parsed"].apply(lambda r: r.get("js_heap_used_mb"))
    df["hardware_concurrency"] = df["resources_parsed"].apply(
        lambda r: r.get("hardware_concurrency")
    )

    return df


df = load_scans()

st.title("👁 Squint — Telemetry Dashboard")
st.caption(
    "Reads directly from the local audit log (squint_audit.db). "
    "Never displays raw PII — only redaction metadata, timings, and resource counters."
)

if df.empty:
    st.info(
        "No scans logged yet. Run a scan from the extension, then refresh this page — "
        "it re-reads the DB every few seconds."
    )
    st.stop()

# ---- top-level metrics ----
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total scans", len(df))
col2.metric("Failed scans", int(df["error"].notna().sum()))
avg_latency = df["total_ms"].dropna().mean()
col3.metric("Avg latency", f"{avg_latency:.0f} ms" if pd.notna(avg_latency) else "—")
col4.metric("Regions redacted", int(df["region_count"].sum()))
noise_scans = df["noise_epsilon"].notna().sum()
col5.metric("Scans with DP noise", f"{noise_scans}/{len(df)}")

st.divider()

# ---- latency panel (15% weight) ----
st.subheader("End-to-end latency")
lat_df = df[["timestamp", "total_ms"]].dropna().sort_values("timestamp")
if not lat_df.empty:
    st.line_chart(lat_df.set_index("timestamp")["total_ms"])
    lc1, lc2, lc3 = st.columns(3)
    lc1.metric("p50", f"{lat_df['total_ms'].median():.0f} ms")
    lc2.metric("p95", f"{lat_df['total_ms'].quantile(0.95):.0f} ms")
    lc3.metric("max", f"{lat_df['total_ms'].max():.0f} ms")
else:
    st.caption("No timing data yet.")

st.divider()

# ---- redaction panel (20% weight — counts/types now; true precision
# needs a labeled ground-truth eval set, noted here rather than faked) ----
st.subheader("Redaction activity")
rc1, rc2 = st.columns([2, 1])

all_region_rows = []
for _, row in df.iterrows():
    for rt in row["region_types_parsed"]:
        all_region_rows.append(rt)

if all_region_rows:
    region_df = pd.DataFrame(all_region_rows)
    agg = (
        region_df.groupby(["type", "reason"])["count"]
        .sum()
        .reset_index()
        .sort_values("count", ascending=False)
    )
    agg["label"] = agg["type"] + " / " + agg["reason"]
    rc1.bar_chart(agg.set_index("label")["count"])
else:
    rc1.caption("No redacted regions logged yet.")

with rc2:
    st.caption("Face detector status (most recent scans)")
    st.dataframe(
        df[["timestamp", "face_detector_status", "face_count"]].head(10),
        hide_index=True,
        use_container_width=True,
    )

st.caption(
    "⚠️ This shows *how much* was redacted and its type breakdown — a full "
    "precision/recall score against ground truth needs a hand-labeled test "
    "page set, which isn't part of this dashboard yet."
)

st.divider()

# ---- resource panel (20% weight) ----
st.subheader("Client resource utilization")
res_df = df[["timestamp", "js_heap_used_mb", "hardware_concurrency"]].dropna(
    subset=["js_heap_used_mb"]
)
if not res_df.empty:
    rcol1, rcol2 = st.columns(2)
    rcol1.line_chart(res_df.set_index("timestamp")["js_heap_used_mb"])
    rcol1.caption("Offscreen document JS heap usage (MB) per scan")
    rcol2.metric(
        "Detected CPU cores",
        int(res_df["hardware_concurrency"].iloc[0])
        if pd.notna(res_df["hardware_concurrency"].iloc[0])
        else "—",
    )
    st.caption(
        "⚠️ Browser extensions can't read true system-wide CPU/GPU utilization — "
        "this is JS heap memory and logical core count only, sampled inside the "
        "offscreen document (the one context in the extension with a real "
        "`performance.memory`). A true CPU/GPU profile would need a native "
        "companion process, which is out of scope for a browser extension."
    )
else:
    st.caption("No resource data yet — `performance.memory` may be unavailable in this browser.")

st.divider()

# ---- privacy / noise panel ----
st.subheader("Differential-privacy noise")
noise_df = df[df["noise_epsilon"].notna()][["timestamp", "noise_epsilon"]]
if not noise_df.empty:
    st.dataframe(noise_df.head(20), hide_index=True, use_container_width=True)
    st.caption(
        f"Epsilon used: {noise_df['noise_epsilon'].iloc[0]} "
        "(pixel-noise budget; see extension/dp-noise.js for the full calibration)."
    )
else:
    st.caption("No scans with noise metadata yet.")

st.divider()

# ---- raw audit trail ----
st.subheader("Recent scan log")
show_cols = [
    "timestamp", "task", "region_count", "face_count",
    "vlm_mode", "vlm_summary", "error",
]
st.dataframe(df[show_cols].head(50), hide_index=True, use_container_width=True)

with st.expander("Full raw row (select an id)"):
    if len(df) > 0:
        selected_id = st.selectbox("Scan id", df["id"].tolist())
        st.json(df[df["id"] == selected_id].iloc[0].drop(
            labels=["region_types_parsed", "timings_parsed", "resources_parsed"]
        ).to_dict())
