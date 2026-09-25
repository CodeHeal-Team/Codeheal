"""
dashboard/app.py
----------------
CodeHeal Streamlit dashboard.

Run with:
    streamlit run dashboard/app.py

from the ``codeheal/`` directory (so that the ``core/`` and ``agents/``
packages are importable).

Responsibilities
----------------
* Sidebar: scenario picker (dropdown + optional custom path) + "Run CodeHeal"
* Trigger: run the pipeline in a background ``threading.Thread``; bridge state
  via ``st.session_state`` using a ``threading.Lock`` for thread-safety
* Render: event timeline, diagnosis panel, side-by-side diff viewer,
  generated-test panel, terminal output expanders, metrics row, reset button
* FAILED banner when ``state.stage == FAILED``
* Auto-refresh with ``st.rerun()`` while the pipeline is in progress
"""

from __future__ import annotations

import sys
import time
import threading
import difflib
from pathlib import Path
from typing import Optional

# ── make sure the codeheal package root is on sys.path ──────────────────────
_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

import streamlit as st

from core.models import PipelineStage, PipelineState, TimelineEvent
from core.pipeline import reset_repo, run_pipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SAMPLES_DIR = _PKG_ROOT / "samples"
_BUNDLED_SCENARIOS: dict[str, str] = {
    "none_bug   — None-guard missing in divide()": str(
        _SAMPLES_DIR / "none_bug"
    ),
    "off_by_one — Off-by-one in last_element()": str(
        _SAMPLES_DIR / "off_by_one"
    ),
    "broken_api — Wrong HTTP method in router": str(
        _SAMPLES_DIR / "broken_api"
    ),
}

_STAGE_LABELS = [
    "Detect",
    "Diagnose",
    "Generate Test",
    "Fix",
    "Verify",
]

# Status → (emoji, CSS colour class name used in styling)
_STATUS_STYLE: dict[str, tuple[str, str]] = {
    "pending":     ("⬜", "#e5e7eb"),
    "in_progress": ("🔧", "#fbbf24"),
    "done":        ("✅", "#22c55e"),
    "failed":      ("❌", "#ef4444"),
}

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------


def _init_session() -> None:
    """Ensure all required session-state keys exist with default values."""
    defaults: dict = {
        "pipeline_state": None,
        "pipeline_running": False,
        "_lock": threading.Lock(),
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Pipeline background thread
# ---------------------------------------------------------------------------


def _state_callback(state: PipelineState) -> None:
    """Called by the pipeline after every stage transition (from worker thread)."""
    with st.session_state["_lock"]:
        st.session_state["pipeline_state"] = state


def _run_pipeline_thread(repo_path: str) -> None:
    """Worker function executed in a background Thread."""
    try:
        state = run_pipeline(repo_path, state_callback=_state_callback)
        with st.session_state["_lock"]:
            st.session_state["pipeline_state"] = state
    finally:
        with st.session_state["_lock"]:
            st.session_state["pipeline_running"] = False


# ---------------------------------------------------------------------------
# Diff utilities
# ---------------------------------------------------------------------------


def _parse_diff_sides(diff_text: str) -> tuple[list[str], list[str]]:
    """
    Split a unified diff into ``(before_lines, after_lines)`` for side-by-side
    display.  Only lines from the diff hunks are extracted; the ``diff --git``
    header and ``@@`` lines are omitted.

    Returns two lists of plain text lines (without the leading +/-/ char).
    """
    before: list[str] = []
    after: list[str] = []
    for line in diff_text.splitlines():
        if line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("diff ") or line.startswith("index ") or line.startswith("@@"):
            continue
        if line.startswith("-"):
            before.append(line[1:])
        elif line.startswith("+"):
            after.append(line[1:])
        else:
            # context line — appears in both
            stripped = line[1:] if line.startswith(" ") else line
            before.append(stripped)
            after.append(stripped)
    return before, after


def _count_files_changed(diff_text: str) -> int:
    """Count how many files are mentioned in a unified diff."""
    return sum(1 for line in diff_text.splitlines() if line.startswith("diff --git "))


# ---------------------------------------------------------------------------
# UI components
# ---------------------------------------------------------------------------


def _render_timeline(timeline: list[TimelineEvent]) -> None:
    """Render the horizontal step indicator for the pipeline stages."""
    st.subheader("Event Timeline")

    # Build a dict of label → latest status for quick lookup
    latest: dict[str, str] = {}
    for event in timeline:
        latest[event.stage] = event.status

    cols = st.columns(len(_STAGE_LABELS))
    for col, label in zip(cols, _STAGE_LABELS):
        status = latest.get(label, "pending")
        emoji, colour = _STATUS_STYLE.get(status, _STATUS_STYLE["pending"])
        with col:
            st.markdown(
                f"""
                <div style="
                    background-color:{colour}22;
                    border:1px solid {colour};
                    border-radius:8px;
                    padding:10px 6px;
                    text-align:center;
                    font-size:0.85rem;
                    ">
                    <div style="font-size:1.4rem;">{emoji}</div>
                    <strong>{label}</strong><br/>
                    <span style="color:{colour};font-size:0.75rem;">{status.replace('_', ' ')}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_diagnosis(state: PipelineState) -> None:
    """Render the Diagnosis panel."""
    d = state.diagnosis
    if not d.root_cause:
        return

    st.subheader("🔍 Diagnosis")

    st.markdown(f"**Root Cause:** {d.root_cause}")
    if d.affected_file:
        lines_str = (
            ", ".join(str(ln) for ln in d.affected_lines) if d.affected_lines else "—"
        )
        st.markdown(f"**Affected File:** `{d.affected_file}` — lines {lines_str}")

    if d.explanation:
        with st.expander("Explanation", expanded=True):
            st.write(d.explanation)

    if d.confidence:
        st.caption(f"Confidence: {d.confidence:.0%}")
        st.progress(min(max(d.confidence, 0.0), 1.0))


def _render_diff(state: PipelineState) -> None:
    """Render the side-by-side diff viewer."""
    if not state.patch or not state.patch.diff:
        return

    st.subheader("📄 Side-by-Side Diff")

    before_lines, after_lines = _parse_diff_sides(state.patch.diff)

    col_before, col_after = st.columns(2)
    with col_before:
        st.markdown("**Before**")
        st.code("\n".join(before_lines), language="python")
    with col_after:
        st.markdown("**After (patched)**")
        st.code("\n".join(after_lines), language="python")


def _render_generated_test(state: PipelineState) -> None:
    """Render the Generated Test panel."""
    gt = state.generated_test
    if not gt.code:
        return

    st.subheader("🧪 Generated Test")

    status_label = (
        "✅ Confirmed failing (bug verified)" if gt.confirmed_failing
        else "⚠️ Could not confirm failure"
    )
    st.caption(status_label)
    if gt.filename:
        st.caption(f"File: `{gt.filename}`")
    st.code(gt.code, language="python")


def _render_terminal_panels(state: PipelineState) -> None:
    """Render collapsible terminal output panels."""
    if state.initial_test_result.stdout or state.initial_test_result.stderr:
        with st.expander("Terminal — Initial Test Run", expanded=False):
            combined = state.initial_test_result.stdout + state.initial_test_result.stderr
            st.code(combined.strip() or "(no output)", language="text")

    if state.final_test_result.stdout or state.final_test_result.stderr:
        with st.expander("Terminal — Final Test Run (after fix)", expanded=False):
            combined = state.final_test_result.stdout + state.final_test_result.stderr
            st.code(combined.strip() or "(no output)", language="text")


def _render_metrics(state: PipelineState) -> None:
    """Render the Before / After / Files Changed metrics row."""
    if state.stage not in (PipelineStage.COMPLETE, PipelineStage.VERIFYING):
        return

    diff_text = state.patch.diff if state.patch else ""
    files_changed = _count_files_changed(diff_text) if diff_text else 0

    before_label = "❌ Failing" if not state.initial_test_result.passed else "✅ Passing"
    after_label = "✅ Passing" if state.final_test_result.passed else "❌ Failing"

    col1, col2, col3 = st.columns(3)
    col1.metric("Before", before_label)
    col2.metric("After", after_label)
    col3.metric("Files Changed", files_changed)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="CodeHeal",
        page_icon="🩺",
        layout="wide",
    )

    _init_session()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🩺 CodeHeal")
        st.caption("AI-powered bug detection & repair")
        st.divider()

        st.subheader("Scenario")
        scenario_display_names = list(_BUNDLED_SCENARIOS.keys())
        selected_display = st.selectbox(
            "Bundled scenario",
            scenario_display_names,
            index=0,
        )
        custom_path = st.text_input(
            "Custom repo path (overrides dropdown)",
            placeholder="/path/to/your/repo",
        )
        repo_path = custom_path.strip() if custom_path.strip() else _BUNDLED_SCENARIOS[selected_display]

        st.divider()

        run_disabled = bool(st.session_state.get("pipeline_running"))
        if st.button("▶ Run CodeHeal", disabled=run_disabled, type="primary", use_container_width=True):
            # Reset repo to HEAD before each run
            reset_repo(repo_path)
            # Clear previous results
            st.session_state["pipeline_state"] = None
            st.session_state["pipeline_running"] = True

            thread = threading.Thread(
                target=_run_pipeline_thread,
                args=(repo_path,),
                daemon=True,
            )
            thread.start()
            st.rerun()

        if st.button("🔄 Reset Repository", use_container_width=True):
            reset_repo(repo_path)
            st.session_state["pipeline_state"] = None
            st.session_state["pipeline_running"] = False
            st.rerun()

    # ── Main area ─────────────────────────────────────────────────────────────
    st.title("🩺 CodeHeal — Automated Bug Repair")

    state: Optional[PipelineState] = st.session_state.get("pipeline_state")
    running: bool = bool(st.session_state.get("pipeline_running"))

    # FAILED banner
    if state is not None and state.stage == PipelineStage.FAILED:
        st.error(f"**Pipeline Failed** — {state.error or 'Unknown error'}", icon="❌")

    # Status bar
    if running:
        st.info("⏳ Pipeline is running… Page refreshes automatically.", icon="🔧")
    elif state is None:
        st.info(
            "Select a scenario in the sidebar and click **▶ Run CodeHeal** to start.",
            icon="👈",
        )

    # Timeline
    if state is not None and state.timeline:
        _render_timeline(state.timeline)
        st.divider()

    # Metrics row (show early if verification done)
    if state is not None:
        _render_metrics(state)

    # Diagnosis
    if state is not None and state.diagnosis and state.diagnosis.root_cause:
        _render_diagnosis(state)
        st.divider()

    # Side-by-side diff
    if state is not None and state.patch and state.patch.diff:
        _render_diff(state)
        st.divider()

    # Generated test
    if state is not None and state.generated_test and state.generated_test.code:
        _render_generated_test(state)
        st.divider()

    # Terminal output
    if state is not None:
        _render_terminal_panels(state)

    # Auto-refresh while pipeline is running
    if running:
        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    main()
