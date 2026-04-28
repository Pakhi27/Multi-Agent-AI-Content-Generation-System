# ============================================================
# 🧠 BWA RESEARCH + IMAGE AGENT  (LangGraph)
# Improvements:
#   - Upgraded to claude-sonnet-4-20250514
#   - Fixed image injection into final markdown
#   - Added retry logic and better error handling
#   - Structured outputs via Pydantic for all nodes
#   - Images saved to a stable absolute path
#   - Sections merged with continuity prompt for complete blogs
# ============================================================

from __future__ import annotations

import base64
import json
import os
import re
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import anthropic
import requests
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

# ── Constants ────────────────────────────────────────────────────────────────

MODEL = "claude-sonnet-4-20250514"          # latest Sonnet 4
FAST_MODEL = "claude-haiku-4-5-20251001"    # fast model for cheap tasks
IMAGES_DIR = Path(__file__).parent / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

client = anthropic.Anthropic()              # reads ANTHROPIC_API_KEY from env


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class SectionTask(BaseModel):
    section_id: str
    heading: str
    goal: str
    word_count: int = 300


class BlogPlan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    tasks: List[SectionTask]


class Evidence(BaseModel):
    title: str
    source: str
    url: str
    snippet: str


class ImageSpec(BaseModel):
    section_id: str
    filename: str
    alt_text: str
    prompt: str          # used for image generation / search query


# ── Graph State ───────────────────────────────────────────────────────────────

class BlogState(TypedDict):
    topic: str
    mode: str                        # "research" | "fast"
    needs_research: bool
    queries: List[str]
    evidence: List[Evidence]
    plan: Optional[BlogPlan]
    sections: List[Dict[str, str]]   # [{"section_id": ..., "content": ...}]
    merged_md: str
    md_with_placeholders: str
    image_specs: List[ImageSpec]
    final: str
    errors: List[str]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _claude(
    system: str,
    user: str,
    model: str = MODEL,
    max_tokens: int = 4096,
    retries: int = 3,
) -> str:
    """Call Claude with retry logic."""
    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return resp.content[0].text
        except anthropic.RateLimitError:
            wait = 2 ** attempt
            time.sleep(wait)
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(1)
    return ""


def _parse_json(text: str) -> Any:
    """Strip markdown fences and parse JSON safely."""
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    return json.loads(text)


def _search_web(query: str, num_results: int = 5) -> List[Dict]:
    """
    Web search via Brave Search API (set BRAVE_API_KEY env var).
    Falls back gracefully if key is missing.
    """
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        return []
    try:
        resp = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"Accept": "application/json", "X-Subscription-Token": api_key},
            params={"q": query, "count": num_results},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json().get("web", {}).get("results", [])
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("description", ""),
            }
            for r in results
        ]
    except Exception:
        return []


def _generate_image(spec: ImageSpec) -> Optional[Path]:
    """
    Generate or fetch an image for a blog section.
    Uses Claude's vision to generate a simple SVG diagram as a fallback
    when no image generation API is available.
    """
    # Try Unsplash free API first (no key needed for demo sizes)
    try:
        url = f"https://source.unsplash.com/800x400/?{requests.utils.quote(spec.prompt)}"
        resp = requests.get(url, timeout=15, allow_redirects=True)
        if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image"):
            out_path = IMAGES_DIR / spec.filename
            out_path.write_bytes(resp.content)
            return out_path
    except Exception:
        pass

    # Fallback: generate a simple SVG diagram via Claude
    try:
        svg_text = _claude(
            system="You are an SVG diagram generator. Output ONLY valid SVG code, no prose, no backticks.",
            user=(
                f"Create a clean, informative SVG diagram (800x400 viewBox) illustrating: {spec.prompt}. "
                "Use a white background, clear labels, and a modern flat style."
            ),
            model=FAST_MODEL,
            max_tokens=2048,
        )
        svg_match = re.search(r"<svg[\s\S]*</svg>", svg_text, re.IGNORECASE)
        if svg_match:
            out_path = IMAGES_DIR / spec.filename.replace(".png", ".svg")
            out_path.write_text(svg_match.group(0), encoding="utf-8")
            return out_path
    except Exception:
        pass

    return None


# ── Graph Nodes ───────────────────────────────────────────────────────────────

def router_node(state: BlogState) -> BlogState:
    """Decide whether the topic needs web research."""
    answer = _claude(
        system="You are a routing classifier. Reply ONLY with JSON: {\"needs_research\": true/false, \"mode\": \"research\"|\"fast\"}",
        user=f"Does writing a technical blog about '{state['topic']}' benefit from current web research? "
             "Answer true if the topic involves recent events, statistics, or rapidly evolving tech.",
        model=FAST_MODEL,
        max_tokens=64,
    )
    try:
        parsed = _parse_json(answer)
        return {**state, **parsed, "errors": state.get("errors", [])}
    except Exception:
        return {**state, "needs_research": True, "mode": "research", "errors": state.get("errors", [])}


def research_node(state: BlogState) -> BlogState:
    """Generate search queries and fetch evidence."""
    if not state.get("needs_research"):
        return state

    # Generate queries
    raw = _claude(
        system='Reply ONLY with a JSON array of 4 search query strings. Example: ["query1","query2"]',
        user=f"Generate web search queries to research: {state['topic']}",
        model=FAST_MODEL,
        max_tokens=256,
    )
    try:
        queries = _parse_json(raw)
    except Exception:
        queries = [state["topic"]]

    evidence: List[Evidence] = []
    for q in queries[:4]:
        results = _search_web(q)
        for r in results:
            if r.get("snippet"):
                evidence.append(
                    Evidence(
                        title=r["title"],
                        source=r["url"].split("/")[2] if r.get("url") else "web",
                        url=r.get("url", ""),
                        snippet=r["snippet"],
                    )
                )

    return {**state, "queries": queries, "evidence": evidence}


def orchestrator_node(state: BlogState) -> BlogState:
    """Create the blog plan with sections."""
    evidence_ctx = ""
    if state.get("evidence"):
        snippets = "\n".join(
            f"- [{e.title}]({e.url}): {e.snippet}" for e in state["evidence"][:8]
        )
        evidence_ctx = f"\n\nAvailable research:\n{snippets}"

    raw = _claude(
        system=textwrap.dedent("""\
            You are a technical blog architect.
            Reply ONLY with valid JSON matching this schema:
            {
              "blog_title": "string",
              "audience": "string",
              "tone": "string",
              "tasks": [
                {"section_id": "s1", "heading": "string", "goal": "string", "word_count": 350}
              ]
            }
            Include 5-7 sections. The first should be an Introduction and the last a Conclusion.
        """),
        user=f"Plan a comprehensive technical blog about: {state['topic']}{evidence_ctx}",
        max_tokens=2048,
    )
    try:
        plan = BlogPlan(**_parse_json(raw))
    except Exception as e:
        # Fallback plan
        plan = BlogPlan(
            blog_title=f"Understanding {state['topic']}",
            audience="developers and technical practitioners",
            tone="clear, informative, and engaging",
            tasks=[
                SectionTask(section_id="s1", heading="Introduction", goal="Introduce the topic", word_count=200),
                SectionTask(section_id="s2", heading="Core Concepts", goal="Explain fundamentals", word_count=400),
                SectionTask(section_id="s3", heading="Deep Dive", goal="Technical details", word_count=500),
                SectionTask(section_id="s4", heading="Practical Examples", goal="Show real use cases", word_count=400),
                SectionTask(section_id="s5", heading="Conclusion", goal="Summarize and call to action", word_count=200),
            ],
        )

    return {**state, "plan": plan}


def worker_node(state: BlogState) -> BlogState:
    """Write each section in parallel (sequential here, but structured for easy parallelization)."""
    plan: BlogPlan = state["plan"]
    evidence = state.get("evidence", [])

    evidence_ctx = ""
    if evidence:
        evidence_ctx = "\n\nUse these references where relevant:\n" + "\n".join(
            f"- {e.title} ({e.source}): {e.snippet}" for e in evidence[:10]
        )

    sections: List[Dict[str, str]] = []
    image_specs: List[ImageSpec] = []

    for i, task in enumerate(plan.tasks):
        system = textwrap.dedent(f"""\
            You are a technical writer producing a section of a blog post.
            Blog title: {plan.blog_title}
            Audience: {plan.audience}
            Tone: {plan.tone}

            Write ONLY the section content in Markdown (no title at top—the heading will be added).
            Target ~{task.word_count} words. Be thorough, concrete, and avoid fluff.
            If relevant, include a code block or formula.
            At the end, if this section benefits from a diagram or illustration,
            add exactly ONE line: IMAGE_PLACEHOLDER::<filename.png>::<alt text>::<image description for generation>
        """)

        content = _claude(
            system=system,
            user=f"Write the '{task.heading}' section.\nGoal: {task.goal}{evidence_ctx}",
            max_tokens=1500,
        )

        # Extract image placeholder if present
        img_match = re.search(
            r"IMAGE_PLACEHOLDER::([^:]+)::([^:]+)::(.+)", content
        )
        if img_match:
            fname, alt, prompt = img_match.group(1), img_match.group(2), img_match.group(3)
            # Sanitize filename
            fname = re.sub(r"[^a-z0-9_\-.]", "_", fname.lower())
            image_specs.append(
                ImageSpec(
                    section_id=task.section_id,
                    filename=fname,
                    alt_text=alt,
                    prompt=prompt,
                )
            )
            # Replace placeholder with markdown image reference
            content = re.sub(
                r"IMAGE_PLACEHOLDER::[^\n]+",
                f"![{alt}](images/{fname})",
                content,
            )

        sections.append({"section_id": task.section_id, "heading": task.heading, "content": content})

    return {**state, "sections": sections, "image_specs": image_specs}


def reducer_node(state: BlogState) -> BlogState:
    """Merge all sections into a single coherent Markdown document."""
    plan: BlogPlan = state["plan"]
    sections = state["sections"]

    # Build raw merged markdown
    parts = [f"# {plan.blog_title}\n"]
    parts.append(f"*Audience: {plan.audience} · Tone: {plan.tone}*\n\n---\n")

    for sec in sections:
        parts.append(f"\n## {sec['heading']}\n\n{sec['content'].strip()}\n")

    merged = "\n".join(parts)

    # Polish pass — fix transitions, remove repetition, ensure completeness
    polished = _claude(
        system=textwrap.dedent("""\
            You are a senior technical editor. You receive a draft blog in Markdown.
            Your job:
            1. Ensure smooth transitions between sections.
            2. Remove any repeated sentences or ideas.
            3. Make sure the Introduction sets context and the Conclusion summarizes properly.
            4. Fix any broken Markdown (unclosed bold, bad headers, etc.).
            5. Keep all ## headings, code blocks, and image references (![...](...)) exactly as-is.
            6. Do NOT add new sections or change the structure.
            7. Return ONLY the improved Markdown, no commentary.
        """),
        user=merged,
        max_tokens=6000,
    )

    return {**state, "merged_md": merged, "final": polished or merged}


def image_node(state: BlogState) -> BlogState:
    """Generate/fetch images and update the final markdown with real paths."""
    image_specs = state.get("image_specs", [])
    final = state.get("final", "")

    for spec in image_specs:
        out_path = _generate_image(spec)
        if out_path:
            # Update any remaining placeholder references (both .png and .svg)
            for ext in [spec.filename, spec.filename.replace(".png", ".svg")]:
                final = final.replace(f"images/{spec.filename}", f"images/{out_path.name}")

    return {**state, "final": final}


# ── Build Graph ───────────────────────────────────────────────────────────────

def _should_research(state: BlogState) -> str:
    return "research" if state.get("needs_research") else "orchestrator"


workflow = StateGraph(BlogState)

workflow.add_node("router", router_node)
workflow.add_node("research", research_node)
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("worker", worker_node)
workflow.add_node("reducer", reducer_node)
workflow.add_node("image", image_node)

workflow.set_entry_point("router")
workflow.add_conditional_edges("router", _should_research, {
    "research": "research",
    "orchestrator": "orchestrator",
})
workflow.add_edge("research", "orchestrator")
workflow.add_edge("orchestrator", "worker")
workflow.add_edge("worker", "reducer")
workflow.add_edge("reducer", "image")
workflow.add_edge("image", END)

app = workflow.compile()
