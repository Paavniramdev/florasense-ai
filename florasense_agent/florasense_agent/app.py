"""
app.py — FloraSense AI Streamlit frontend (Week 4).

Drop this file directly into the same folder as graph.py, tools.py, and
main.py (i.e. your florasense_agent folder) — it imports from them directly,
which sidesteps path issues entirely.

Also copy analytics_db.py into that same folder.

Run with:
    streamlit run app.py
"""
import json
import os
import tempfile
import time

import streamlit as st
from langchain_core.messages import HumanMessage

import analytics_db as db
from graph import build_agent
from main import extract_text

st.set_page_config(page_title="FloraSense AI", page_icon="🌿", layout="wide")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;1,9..144,500&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --paper: #F6F3EA;
    --paper-deep: #EDE7D6;
    --ink: #21301D;
    --ink-soft: #4A5844;
    --fern: #3C6E47;
    --moss: #8A9A5B;
    --ochre: #B98A2D;
    --rust: #9C4A32;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--ink); }

h1, h2, h3 { font-family: 'Fraunces', serif; color: var(--ink); letter-spacing: -0.01em; }

.fs-header {
    display: flex; align-items: baseline; gap: 0.6rem;
    border-bottom: 1.5px solid var(--ink); padding-bottom: 0.6rem; margin-bottom: 0.3rem;
}
.fs-header h1 { font-size: 2.1rem; font-weight: 600; margin: 0; }
.fs-header .fs-latin { font-family: 'Fraunces', serif; font-style: italic; font-weight: 500; color: var(--fern); font-size: 1.15rem; }
.fs-tagline { font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; color: var(--ink-soft);
    text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.4rem; margin-bottom: 1.4rem; }

/* --- Signature element: the specimen identification card --- */
.fs-specimen {
    position: relative; border: 1px solid var(--ink); border-radius: 3px;
    padding: 1.3rem 1.5rem; margin: 1rem 0 1.2rem 0; background: var(--paper-deep);
}
.fs-specimen::before, .fs-specimen::after,
.fs-specimen .fs-corner-br::before, .fs-specimen .fs-corner-br::after {
    content: ""; position: absolute; width: 10px; height: 10px; border-color: var(--fern);
}
.fs-specimen::before { top: -1px; left: -1px; border-top: 2px solid var(--fern); border-left: 2px solid var(--fern); }
.fs-specimen::after { top: -1px; right: -1px; border-top: 2px solid var(--fern); border-right: 2px solid var(--fern); }
.fs-eyebrow {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; text-transform: uppercase;
    letter-spacing: 0.12em; color: var(--ink-soft); margin-bottom: 0.35rem;
}
.fs-species-name { font-family: 'Fraunces', serif; font-style: italic; font-weight: 500;
    font-size: 1.5rem; color: var(--ink); margin: 0 0 0.5rem 0; }
.fs-confidence-row { display: flex; align-items: center; gap: 0.8rem; }
.fs-confidence-value { font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem; font-weight: 500; }
.fs-confidence-bar { flex: 1; height: 6px; background: #DDD5BE; border-radius: 3px; overflow: hidden; }
.fs-confidence-fill { height: 100%; border-radius: 3px; }
.fs-badge { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; text-transform: uppercase;
    letter-spacing: 0.08em; padding: 0.15rem 0.55rem; border-radius: 3px; border: 1px solid; }
.fs-badge-confident { color: var(--fern); border-color: var(--fern); background: rgba(60,110,71,0.08); }
.fs-badge-uncertain { color: var(--ochre); border-color: var(--ochre); background: rgba(185,138,45,0.1); }

/* --- Buttons --- */
.stButton > button {
    font-family: 'Inter', sans-serif; font-weight: 500; border-radius: 4px !important;
    border: 1px solid var(--ink) !important;
}
.stButton > button[kind="primary"] {
    background-color: var(--fern) !important; border-color: var(--fern) !important;
}

/* --- Metric cards (analytics dashboard) --- */
[data-testid="stMetric"] {
    background: var(--paper-deep); border: 1px solid var(--ink); border-radius: 3px;
    padding: 0.9rem 1rem 0.7rem 1rem;
}
[data-testid="stMetricLabel"] { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.72rem !important;
    text-transform: uppercase; letter-spacing: 0.08em; color: var(--ink-soft) !important; }
[data-testid="stMetricValue"] { font-family: 'Fraunces', serif !important; color: var(--ink) !important; }

/* --- Tabs --- */
.stTabs [data-baseweb="tab"] { font-family: 'Inter', sans-serif; font-weight: 500; }
.stTabs [aria-selected="true"] { color: var(--fern) !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--fern) !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
db.init_db()

# Note: this version talks to your FastAPI backend (localhost:8000) for
# classification, so there's no in-process model to load here — that's the
# florasense_deploy build (for Hugging Face Spaces), not this one.


@st.cache_resource(show_spinner=False)
def get_agent():
    return build_agent()


def run_agent_and_log(question: str, image_path: str = None, location: str = None):
    """Runs the agent, streaming tool calls, and logs the interaction for analytics.
    Returns (final_answer_text, tool_trace, predicted_species, confidence, is_confident, interaction_id).
    """
    agent = get_agent()

    prompt_parts = [question]
    if image_path:
        prompt_parts.append(f"[Attached image file path: {image_path}]")
    if location:
        prompt_parts.append(f"[User's location: {location}]")
    user_message = HumanMessage(content="\n".join(prompt_parts))

    tool_trace = []
    predicted_species, confidence, is_confident = None, None, None

    start = time.time()
    final_state = None
    for step in agent.stream({"messages": [user_message]}, stream_mode="values"):
        final_state = step
        last = step["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                tool_trace.append({"type": "call", "name": tc["name"], "args": tc["args"]})
        elif getattr(last, "type", None) == "tool":
            tool_trace.append({"type": "result", "name": last.name, "content": extract_text(last.content)})
            if last.name == "classify_flower":
                try:
                    parsed = json.loads(last.content)
                    if "predicted_species" in parsed:
                        predicted_species = parsed["predicted_species"]
                        confidence = parsed["confidence"]
                        is_confident = parsed["is_confident"]
                except (json.JSONDecodeError, TypeError):
                    pass
    latency = time.time() - start

    final_answer = extract_text(final_state["messages"][-1].content)
    tools_used = [t["name"] for t in tool_trace if t["type"] == "call"]

    interaction_id = db.log_interaction(
        question=question,
        had_image=bool(image_path),
        predicted_species=predicted_species,
        confidence=confidence,
        is_confident=is_confident,
        tools_used=tools_used,
        latency_seconds=latency,
        answer_preview=final_answer,
    )

    return final_answer, tool_trace, predicted_species, confidence, is_confident, interaction_id


# --- UI -----------------------------------------------------------------

ask_tab, analytics_tab = st.tabs(["🌿 Ask FloraSense", "📊 Analytics Dashboard"])

with ask_tab:
    st.markdown(
        '<div class="fs-header"><h1>FloraSense</h1><span class="fs-latin">Herbarium Intelligentia</span></div>'
        '<div class="fs-tagline">Field identification &middot; grounded knowledge &middot; agentic reasoning</div>',
        unsafe_allow_html=True,
    )

    if not os.getenv("GOOGLE_API_KEY"):
        st.error(
            "GOOGLE_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey, "
            "then set it before running `streamlit run app.py`."
        )
        st.stop()

    col1, col2 = st.columns([1, 1.3])

    with col1:
        uploaded_image = st.file_uploader("Upload a flower photo (optional)", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_image:
            st.image(uploaded_image, caption="Uploaded image", use_container_width=True)

    with col2:
        question = st.text_area(
            "Your question",
            placeholder="e.g. What's wrong with my plant's leaves? / What pollinators visit this flower?",
            height=100,
        )
        location = st.text_input("Location (optional — enables weather-aware answers)", placeholder="e.g. Patiala, Punjab")
        submit = st.button("Ask FloraSense", type="primary", use_container_width=True)

    if submit:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            image_path = None
            if uploaded_image is not None:
                suffix = os.path.splitext(uploaded_image.name)[1] or ".jpg"
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(uploaded_image.getvalue())
                tmp.close()
                image_path = tmp.name

            with st.spinner("Thinking through your question..."):
                try:
                    answer, tool_trace, species, confidence, is_confident, interaction_id = run_agent_and_log(
                        question, image_path, location or None
                    )
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
                    st.stop()

            st.session_state["last_interaction_id"] = interaction_id

            if species:
                conf_pct = confidence * 100
                bar_color = "var(--fern)" if is_confident else "var(--ochre)"
                badge_class = "fs-badge-confident" if is_confident else "fs-badge-uncertain"
                badge_label = "Confident ID" if is_confident else "Uncertain — verify"
                st.markdown(
                    f"""
                    <div class="fs-specimen">
                        <div class="fs-eyebrow">Specimen identified</div>
                        <div class="fs-species-name">{species}</div>
                        <div class="fs-confidence-row">
                            <span class="fs-confidence-value" style="color:{bar_color}">{conf_pct:.1f}%</span>
                            <div class="fs-confidence-bar">
                                <div class="fs-confidence-fill" style="width:{conf_pct}%; background:{bar_color};"></div>
                            </div>
                            <span class="fs-badge {badge_class}">{badge_label}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("### Answer")
            st.markdown(answer)

            with st.expander(f"🔧 Agent reasoning trace ({len([t for t in tool_trace if t['type']=='call'])} tool calls)"):
                for step in tool_trace:
                    if step["type"] == "call":
                        st.markdown(f"**→ called `{step['name']}`** with `{step['args']}`")
                    else:
                        st.code(step["content"][:500], language="json")

            st.markdown("---")
            fb_col1, fb_col2, _ = st.columns([1, 1, 6])
            with fb_col1:
                if st.button("👍 Helpful"):
                    db.log_feedback(interaction_id, positive=True)
                    st.toast("Thanks for the feedback!")
            with fb_col2:
                if st.button("👎 Not helpful"):
                    db.log_feedback(interaction_id, positive=False)
                    st.toast("Thanks — noted for improvement.")

with analytics_tab:
    st.markdown(
        '<div class="fs-header"><h1>Field Notes</h1><span class="fs-latin">Collection Statistics</span></div>'
        '<div class="fs-tagline">Usage &middot; accuracy &middot; feedback</div>',
        unsafe_allow_html=True,
    )
    stats = db.get_summary_stats()

    if stats["total_queries"] == 0:
        st.info("No queries logged yet — ask FloraSense something first!")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total queries", stats["total_queries"])
        m2.metric("Avg latency", f"{stats['avg_latency']}s")
        m3.metric("Avg classifier confidence", f"{stats['avg_confidence']:.1%}" if stats["avg_confidence"] else "—")
        m4.metric(
            "Positive feedback rate",
            f"{stats['positive_feedback_rate']:.0%}" if stats["positive_feedback_rate"] is not None else "no feedback yet",
        )

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Most queried species")
            if stats["top_species"]:
                st.bar_chart({species: count for species, count in stats["top_species"]})
            else:
                st.caption("No classified images yet.")

        with col_b:
            st.subheader("Tool usage")
            if stats["tool_usage"]:
                st.bar_chart(stats["tool_usage"])
            else:
                st.caption("No tool usage recorded yet.")

        st.subheader("Recent queries")
        recent = db.get_recent_interactions(limit=20)
        for row in recent:
            conf_str = f"{row['confidence']:.0%}" if row["confidence"] is not None else "—"
            fb_str = {1: "👍", 0: "👎", None: ""}[row["feedback"]]
            st.markdown(
                f"**{row['timestamp'][:19]}** — _{row['question'][:80]}_  "
                f"→ {row['predicted_species'] or 'n/a'} ({conf_str}) · "
                f"{row['latency_seconds']:.1f}s {fb_str}"
            )
