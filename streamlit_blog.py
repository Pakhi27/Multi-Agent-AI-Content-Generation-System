# ============================================================
# ✨ AI BLOG GENERATOR — PREMIUM UI
# ============================================================

from __future__ import annotations

import json
import re
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from bwa_research_image import app


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="BlogForge AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Premium CSS — Dark editorial theme
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root tokens ── */
:root {
    --bg:        #0d0f12;
    --bg-card:   #13161b;
    --bg-raised: #1a1e25;
    --border:    #242830;
    --border-hi: #2e3440;
    --accent:    #e8c97a;
    --accent2:   #7a9ee8;
    --text:      #e8eaf0;
    --text-dim:  #7a7f8e;
    --text-muted:#4a4f5e;
    --success:   #5cb88a;
    --error:     #e87a7a;
    --radius:    12px;
    --radius-lg: 20px;
}

/* ── Full app background ── */
.stApp, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif;
}

/* ── Remove default Streamlit chrome ── */
[data-testid="stHeader"],
[data-testid="stToolbar"],
footer { display: none !important; }

/* ── Main content area ── */
.block-container {
    padding: 2.5rem 3rem 4rem !important;
    max-width: 1400px !important;
}

/* ── Hero header ── */
.hero-wrap {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    margin-bottom: 2.5rem;
    padding-bottom: 2rem;
    border-bottom: 1px solid var(--border);
}
.hero-badge {
    background: linear-gradient(135deg, #e8c97a22, #7a9ee822);
    border: 1px solid var(--accent);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 22px;
    line-height: 1;
}
.hero-title {
    font-family: 'Instrument Serif', Georgia, serif;
    font-size: 2.4rem;
    font-weight: 400;
    font-style: italic;
    color: var(--text);
    margin: 0;
    line-height: 1.1;
}
.hero-title span { color: var(--accent); }
.hero-sub {
    font-size: 0.875rem;
    color: var(--text-dim);
    margin-top: 4px;
    letter-spacing: 0.02em;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 2rem 1.4rem !important;
}

/* Sidebar section labels */
.sidebar-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 1.6rem 0 0.6rem;
}

/* ── Inputs ── */
[data-testid="stTextArea"] textarea,
[data-testid="stDateInput"] input {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 12px 14px !important;
    transition: border-color 0.2s;
}
[data-testid="stTextArea"] textarea:focus,
[data-testid="stDateInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px #e8c97a18 !important;
}
[data-testid="stTextArea"] label,
[data-testid="stDateInput"] label,
[data-testid="stSelectbox"] label {
    color: var(--text-dim) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
}

/* ── Generate button ── */
[data-testid="stButton"] > button[kind="primary"],
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #e8c97a, #d4a843) !important;
    color: #0d0f12 !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.03em !important;
    padding: 12px !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px #e8c97a30 !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px #e8c97a45 !important;
}

/* ── Download buttons ── */
[data-testid="stDownloadButton"] > button {
    background: var(--bg-raised) !important;
    color: var(--text) !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: var(--radius) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    padding: 10px 16px !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
[data-testid="stDownloadButton"] > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* ── Tabs ── */
[data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
    padding-bottom: 0 !important;
    margin-bottom: 2rem !important;
}
[data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    color: var(--text-muted) !important;
    padding: 12px 20px !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    transition: all 0.2s !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
    background: transparent !important;
}
[data-baseweb="tab"]:hover {
    color: var(--text) !important;
}

/* ── Cards / panels ── */
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.stat-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 6px;
}
.stat-value {
    font-size: 1.05rem;
    color: var(--text);
    font-weight: 400;
}

/* ── Blog article render ── */
.blog-wrap {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 3rem 3.5rem;
    line-height: 1.85;
    max-width: 780px;
    margin: 0 auto 2rem;
}
.blog-wrap h1 {
    font-family: 'Instrument Serif', Georgia, serif !important;
    font-size: 2.5rem !important;
    font-weight: 400 !important;
    color: var(--text) !important;
    border: none !important;
    padding: 0 !important;
    margin-bottom: 0.5rem !important;
}
.blog-wrap h2 {
    font-family: 'Instrument Serif', Georgia, serif !important;
    font-size: 1.6rem !important;
    font-weight: 400 !important;
    font-style: italic !important;
    color: var(--accent) !important;
    margin-top: 2.5rem !important;
    border: none !important;
}
.blog-wrap h3 {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: var(--text) !important;
    margin-top: 1.8rem !important;
}
.blog-wrap p {
    color: #c8cad4 !important;
    font-size: 1rem !important;
}
.blog-wrap code {
    font-family: 'JetBrains Mono', monospace !important;
    background: var(--bg-raised) !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: 5px !important;
    padding: 2px 7px !important;
    font-size: 0.85em !important;
    color: var(--accent2) !important;
}
.blog-wrap pre {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: var(--radius) !important;
    padding: 1.2rem 1.4rem !important;
    overflow-x: auto !important;
}
.blog-wrap blockquote {
    border-left: 3px solid var(--accent) !important;
    padding: 0.4rem 0 0.4rem 1.2rem !important;
    color: var(--text-dim) !important;
    font-style: italic !important;
    margin: 1.5rem 0 !important;
    background: #e8c97a08 !important;
    border-radius: 0 8px 8px 0 !important;
}

/* ── Plan view ── */
.plan-header {
    background: linear-gradient(135deg, var(--bg-card), var(--bg-raised));
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
}
.plan-title {
    font-family: 'Instrument Serif', Georgia, serif;
    font-size: 1.8rem;
    color: var(--text);
    font-style: italic;
    margin-bottom: 1rem;
}
.plan-meta {
    display: flex;
    gap: 2rem;
    flex-wrap: wrap;
}
.plan-meta-item { font-size: 0.85rem; color: var(--text-dim); }
.plan-meta-item span { color: var(--text); font-weight: 500; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: var(--radius) !important;
    overflow: hidden !important;
}

/* ── Alert/info boxes ── */
[data-testid="stAlert"] {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: var(--radius) !important;
    color: var(--text-dim) !important;
}

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: var(--accent) !important; }

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* ── Image gallery ── */
.img-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    transition: border-color 0.2s;
    margin-bottom: 1rem;
}
.img-card:hover { border-color: var(--accent); }
.img-caption {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-muted);
    padding: 8px 12px;
    background: var(--bg-raised);
}

/* ── Log viewer ── */
.log-wrap {
    background: var(--bg-raised);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-dim);
    padding: 1.2rem;
    max-height: 600px;
    overflow-y: auto;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 5rem 2rem;
    color: var(--text-muted);
}
.empty-state .icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    opacity: 0.4;
}
.empty-state h3 {
    font-family: 'Instrument Serif', serif;
    font-size: 1.4rem;
    font-weight: 400;
    font-style: italic;
    color: var(--text-dim);
    margin-bottom: 0.5rem;
}
.empty-state p {
    font-size: 0.875rem;
    color: var(--text-muted);
    max-width: 320px;
    margin: 0 auto;
}

/* ── Sidebar logo area ── */
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.8rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
}
.sidebar-logo-mark {
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, #e8c97a, #d4a843);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    color: #0d0f12;
    font-weight: 700;
    flex-shrink: 0;
}
.sidebar-logo-text {
    font-weight: 600;
    font-size: 1rem;
    color: var(--text);
    letter-spacing: -0.01em;
}
.sidebar-logo-sub {
    font-size: 0.72rem;
    color: var(--text-muted);
    letter-spacing: 0.04em;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Helpers
# -----------------------------
def safe_slug(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return s or "blog"


def bundle_zip(md_text: str, md_filename: str, images_dir: Path) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(md_filename, md_text.encode("utf-8"))
        if images_dir.exists():
            for p in images_dir.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(p))
    return buf.getvalue()


# -----------------------------
# Hero Header
# -----------------------------
st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">✦</div>
    <div>
        <div class="hero-title">BlogForge <span>AI</span></div>
        <div class="hero-sub">Research-backed technical blogs — generated end-to-end with multi-agent AI</div>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-mark">✦</div>
        <div>
            <div class="sidebar-logo-text">BlogForge</div>
            <div class="sidebar-logo-sub">AI &nbsp;·&nbsp; v2.0</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Topic</div>', unsafe_allow_html=True)
    topic = st.text_area(
        "Topic",
        label_visibility="collapsed",
        placeholder="e.g. How RAG improves LLM accuracy in production...",
        height=130
    )

    st.markdown('<div class="sidebar-label">As-of Date</div>', unsafe_allow_html=True)
    as_of = st.date_input("Date", value=date.today(), label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("✦  Generate Blog", use_container_width=True)

    st.markdown('<div class="sidebar-label">Past Blogs</div>', unsafe_allow_html=True)
    files = list(Path(".").glob("*.md"))

    if files:
        selected = st.selectbox("Select blog", files, label_visibility="collapsed")
        if st.button("Load Blog", use_container_width=True):
            st.session_state["last_out"] = {
                "final": selected.read_text(encoding="utf-8"),
                "plan": None,
                "evidence": [],
                "image_specs": []
            }
    else:
        st.markdown(
            '<p style="font-size:0.8rem;color:#4a4f5e;margin-top:0.3rem;">No saved blogs yet</p>',
            unsafe_allow_html=True
        )

    # Pipeline steps indicator
    st.markdown('<div class="sidebar-label">Pipeline</div>', unsafe_allow_html=True)
    pipeline_steps = [
        ("Router", "routes topic to research mode"),
        ("Research", "fetches web evidence"),
        ("Orchestrator", "builds structured plan"),
        ("Workers", "writes each section"),
        ("Reducer", "merges + adds images"),
    ]
    pipeline_html = ""
    for name, desc in pipeline_steps:
        pipeline_html += f"""
        <div style="display:flex;align-items:flex-start;gap:8px;margin-bottom:10px;">
            <div style="width:6px;height:6px;border-radius:50%;background:#2e3440;flex-shrink:0;margin-top:5px;"></div>
            <div>
                <span style="font-size:0.8rem;color:#7a7f8e;font-weight:500;">{name}</span>
                <br><span style="font-size:0.72rem;color:#4a4f5e;">{desc}</span>
            </div>
        </div>"""
    st.markdown(pipeline_html, unsafe_allow_html=True)


# -----------------------------
# Tabs
# -----------------------------
tab_preview, tab_plan, tab_evidence, tab_images, tab_logs = st.tabs(
    ["  Blog Preview  ", "  Content Plan  ", "  Research  ", "  Images  ", "  Debug  "]
)

# -----------------------------
# Session State Init
# -----------------------------
if "last_out" not in st.session_state:
    st.session_state["last_out"] = None

# -----------------------------
# Run
# -----------------------------
if run_btn:
    if not topic.strip():
        st.warning("Please enter a topic before generating.")
        st.stop()

    inputs = {
        "topic": topic,
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
    }

    progress = st.progress(0)
    status = st.empty()

    status.markdown(
        '<p style="font-size:0.85rem;color:#7a7f8e;">⚙️ Initialising pipeline…</p>',
        unsafe_allow_html=True
    )

    try:
        with st.spinner("Generating your blog — this takes 1–2 minutes…"):
            out = app.invoke(inputs)

        st.session_state["last_out"] = out
        progress.progress(1.0)
        status.markdown(
            '<p style="font-size:0.875rem;color:#5cb88a;font-weight:600;">✓ Blog generated successfully</p>',
            unsafe_allow_html=True
        )

    except Exception as e:
        progress.empty()
        status.markdown(
            f'<p style="font-size:0.875rem;color:#e87a7a;">✕ Error: {e}</p>',
            unsafe_allow_html=True
        )

# -----------------------------
# Render Output
# -----------------------------
out = st.session_state.get("last_out")

if out:
    final_md = out.get("final", "")

    # ── Blog Preview ──
    with tab_preview:
        if final_md:
            st.markdown('<div class="blog-wrap">', unsafe_allow_html=True)

            parts = re.split(
                r'(!\[.*?\]\(images/[^\)]+\)|<img\s+src="images/[^"]+"[^>]*>)',
                final_md
            )

            for part in parts:
                if part.startswith('!['):
                    match = re.search(r'\((images/[^\)]+)\)', part)
                    if match:
                        img_path = match.group(1)
                        if Path(img_path).exists():
                            st.image(img_path, use_container_width=True)
                        else:
                            st.markdown(
                                f'<p style="font-size:0.8rem;color:#4a4f5e;font-style:italic;">Image not found: {img_path}</p>',
                                unsafe_allow_html=True
                            )
                elif part.startswith('<img '):
                    match = re.search(r'src="(images/[^"]+)"', part)
                    if match:
                        img_path = match.group(1)
                        if Path(img_path).exists():
                            st.image(img_path, use_container_width=True)
                        else:
                            st.markdown(
                                f'<p style="font-size:0.8rem;color:#4a4f5e;font-style:italic;">Image not found: {img_path}</p>',
                                unsafe_allow_html=True
                            )
                else:
                    if part.strip():
                        st.markdown(part, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            title = "blog"
            if out.get("plan"):
                title = out["plan"].blog_title
            filename = f"{safe_slug(title)}.md"

            st.markdown('<div style="max-width:780px;margin:0 auto;margin-top:1rem;">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇  Download Markdown",
                    final_md,
                    file_name=filename
                )
            with col2:
                st.download_button(
                    "⬇  Download Bundle (.zip)",
                    bundle_zip(final_md, filename, Path("images")),
                    file_name=f"{safe_slug(title)}.zip"
                )
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="empty-state">
                <div class="icon">📄</div>
                <h3>No blog generated yet</h3>
                <p>Enter a topic in the sidebar and click Generate to create your first post.</p>
            </div>
            """, unsafe_allow_html=True)

    # ── Plan ──
    with tab_plan:
        plan = out.get("plan")

        if plan:
            if hasattr(plan, "model_dump"):
                plan = plan.model_dump()

            st.markdown(f"""
            <div class="plan-header">
                <div class="plan-title">{plan.get("blog_title", "Untitled")}</div>
                <div class="plan-meta">
                    <div class="plan-meta-item">Audience &nbsp;<span>{plan.get("audience", "—")}</span></div>
                    <div class="plan-meta-item">Tone &nbsp;<span>{plan.get("tone", "—")}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            tasks = plan.get("tasks", [])
            if tasks:
                df = pd.DataFrame(tasks)
                st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="icon">🧩</div>
                <h3>No plan available</h3>
                <p>The content plan will appear here after generation.</p>
            </div>
            """, unsafe_allow_html=True)

    # ── Research / Evidence ──
    with tab_evidence:
        evidence = out.get("evidence", [])

        if evidence:
            rows = []
            for e in evidence:
                if hasattr(e, "model_dump"):
                    e = e.model_dump()
                rows.append({
                    "Title": e.get("title"),
                    "Source": e.get("source"),
                    "URL": e.get("url")
                })
            df_ev = pd.DataFrame(rows)
            st.dataframe(df_ev, use_container_width=True, hide_index=True)

            st.markdown(
                f'<p style="font-size:0.8rem;color:#4a4f5e;margin-top:0.6rem;">{len(rows)} sources retrieved</p>',
                unsafe_allow_html=True
            )
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="icon">🔎</div>
                <h3>No research data</h3>
                <p>Research sources will appear here when the blog uses web research mode.</p>
            </div>
            """, unsafe_allow_html=True)

    # ── Images ──
    with tab_images:
        images_dir = Path("images")
        image_specs = out.get("image_specs", [])

        if image_specs:
            cols = st.columns(3, gap="medium")
            for i, spec in enumerate(image_specs):
                filename = spec.get("filename")
                if filename:
                    img_path = images_dir / filename
                    if img_path.exists():
                        with cols[i % 3]:
                            st.markdown('<div class="img-card">', unsafe_allow_html=True)
                            st.image(str(img_path), use_container_width=True)
                            st.markdown(
                                f'<div class="img-caption">{filename}</div>',
                                unsafe_allow_html=True
                            )
                            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="icon">🖼</div>
                <h3>No images generated</h3>
                <p>AI-generated images for this blog will appear here.</p>
            </div>
            """, unsafe_allow_html=True)

    # ── Debug Logs ──
    with tab_logs:
        def serialize(obj):
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            if isinstance(obj, list):
                return [serialize(x) for x in obj]
            if isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            return obj

        safe_out = serialize(out)
        log_str = json.dumps(safe_out, indent=2)

        st.markdown(
            f'<div class="log-wrap"><pre style="margin:0;white-space:pre-wrap;word-break:break-word;">{log_str}</pre></div>',
            unsafe_allow_html=True
        )

else:
    # ── No output yet ──
    with tab_preview:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">✦</div>
            <h3>Ready to generate</h3>
            <p>Enter your topic in the sidebar and click <strong style="color:#e8c97a;">Generate Blog</strong> to get started.</p>
        </div>
        """, unsafe_allow_html=True)
