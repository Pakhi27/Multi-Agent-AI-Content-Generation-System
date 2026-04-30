from __future__ import annotations

import operator
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated
from unittest import result

from dotenv import load_dotenv
from pydantic import BaseModel, Field,field_validator

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
import os
import urllib.parse
import requests
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# -----------------------------
# 1) Schemas
# -----------------------------
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(
        ...,
        description="One sentence describing what the reader should be able to do/understand after this section.",
    )
    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=4,
        description="3–4 concrete, non-overlapping subpoints to cover in this section.",
    )
    target_words: int = Field(..., description="Target word count for this section (150–350).")
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task] = Field(..., min_length=4, max_length=6)


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = Field(default=None, max_length=200)
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"] = "closed_book"
    queries: List[str] = Field(default_factory=list)

    @field_validator("needs_research", mode="before")
    @classmethod
    def coerce_bool(cls, v):
        if isinstance(v, str):
            return v.strip().lower() not in ("false", "0", "no", "")
        return v


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


# -----------------------------
# 2) State
# -----------------------------
class State(TypedDict):
    topic: str

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)

    # reducer/image
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]

    final: str



# -----------------------------
# 2b) Worker subgraph state
# Note: Send() passes a dict that becomes the node's state.
# We define a separate typed dict for the worker payload.
# -----------------------------
class WorkerState(TypedDict):
    task: dict
    topic: str
    mode: str
    plan: dict
    evidence: List[dict]


load_dotenv()

llm_strong = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    max_retries=5
)

llm_fast = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    max_retries=5
)

llm_worker = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    max_retries=5
)

# -----------------------------
# 3) Router
# -----------------------------
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false):
  Evergreen topics where correctness does not depend on recent facts (concepts, fundamentals).
- hybrid (needs_research=true):
  Mostly evergreen but needs up-to-date examples/tools/models to be useful.
- open_book (needs_research=true):
  Mostly volatile: weekly roundups, "this week", "latest", rankings, pricing, policy/regulation.

If needs_research=true, include 3-5 specific search queries.

You MUST always respond with a JSON object containing ALL THREE of these keys:
{
  "needs_research": true or false,
  "mode": "closed_book" or "hybrid" or "open_book",
  "queries": []
}
Never omit any key. Use JSON true/false (not strings).
"""


def router_node(state: State) -> dict:
    topic = state["topic"]
    decider = llm_strong.with_structured_output(RouterDecision, method="json_mode")
    decision = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {topic}"),
        ]
    )
    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
    }


def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"


# -----------------------------
# 4) Research
# -----------------------------
def _clean_snippet(text: str, max_len: int = 200) -> str:
    if not text:
        return ""
    # Strip markdown headings, links, code fences
    text = re.sub(r"#{1,6}\s+", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]

def _tavily_search(query: str, max_results: int = 3) -> List[dict]:
    tool = TavilySearchResults(max_results=max_results)
    response = tool.invoke({"query": query})
    results = response if isinstance(response, list) else response.get("results", [])

    normalized: List[dict] = []
    for r in results or []:
        if isinstance(r, dict):
            normalized.append({
                "title": (r.get("title") or "")[:80],
                "url": r.get("url") or "",
                 "snippet": _clean_snippet(r.get("content") or r.get("snippet") or ""),
                "published_at": r.get("published_date") or r.get("published_at"),
                "source": r.get("source"),
            })
    return normalized


RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.

Given raw web search results, produce a deduplicated list of EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Strongly prefer official documentation, research papers, framework docs, and reputable engineering blogs.
- Avoid low-authority recap sites unless nothing better exists.
- If a published date is explicitly present in the result payload, keep it as YYYY-MM-DD.
  If missing or unclear, set published_at=null. Do NOT guess.
- Keep snippets short.
- Deduplicate by URL.

Respond in JSON format.
"""


def research_node(state: State) -> dict:
    queries = (state.get("queries") or [])[:3]   # max 3 queries
    raw_results: List[dict] = []
    for q in queries:
        raw_results.extend(_tavily_search(q, max_results=3))

    if not raw_results:
        return {"evidence": []}

    # Hard cap at 10 results total before sending to LLM
    raw_results = raw_results[:10]
    
    formatted_results = "\n\n".join(
    [
        f"Title: {r['title']}\nURL: {r['url']}\nPublished: {r.get('published_at')}\nSnippet: {r.get('snippet')}"
        for r in raw_results
    ]
)
    extractor = llm_fast.with_structured_output(EvidencePack, method="json_mode")
    pack = extractor.invoke([
        SystemMessage(content=RESEARCH_SYSTEM),
        HumanMessage(content=f"Raw search results:\n\n{formatted_results}"),
    ])

    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e

    return {"evidence": list(dedup.values())}

# -----------------------------
# 5) Orchestrator
# -----------------------------
ORCH_SYSTEM = """You are a technical blog planner. Produce a concise outline.

Rules:
- Create EXACTLY 4–6 sections (no more than 6).
- Each section: goal (1 sentence), 3–4 bullets, target_words 150–350.
- Include at least 1 section with requires_code=True.
- closed_book: evergreen content only.
- hybrid: use evidence for examples; set requires_citations=True on those sections.
- open_book: blog_kind="news_roundup", summarize events with evidence URLs.

Output must strictly match the Plan schema. Respond in JSON format.
"""


def orchestrator_node(state: State) -> dict:
    planner = llm_strong.with_structured_output(Plan, method="json_mode")
    evidence = state.get("evidence") or []
    mode = state.get("mode") or "closed_book"

    plan = planner.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n\n"
                    f"Evidence (ONLY use for fresh claims; may be empty):\n"
                    f"{[e.model_dump() for e in evidence][:16]}"
                )
            ),
        ]
    )
    return {"plan": plan}


# -----------------------------
# 6) Fanout
# -----------------------------
def fanout(state: State):
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence") or []],
            },
        )
        for task in state["plan"].tasks
    ]


# -----------------------------
# 7) Worker
# Use WorkerState signature so keys are accessed correctly.
# -----------------------------
WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Hard constraints:
- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).
- Stay close to Target words (±15%).
- Output ONLY the section content in Markdown (no blog title H1, no extra commentary).
- Start with a '## <Section Title>' heading.

Scope guard:
- If blog_kind == "news_roundup": do NOT turn this into a tutorial/how-to guide.
  Focus on summarizing events and implications.

Grounding policy:
- If mode == open_book:
  - Do NOT introduce any specific event/company/model/funding/policy claim unless it is supported by provided Evidence URLs.
  - For each event claim, attach a source as a Markdown link using a relevant word or domain name as the text. Do NOT use the word "Source". Example: ([link](URL)) or ([domain.com](URL)).
  - Only use URLs provided in Evidence. If not supported, write: "Not found in provided sources."
- If requires_citations == true:
  - For outside-world claims, cite Evidence URLs the same way.
- Evergreen reasoning is OK without citations unless requires_citations is true.

Code:
- If requires_code == true, include at least one minimal, correct code snippet relevant to the bullets.

Style:
- Short paragraphs, bullets where helpful, code fences for code.
- Avoid fluff/marketing. Be precise and implementation-oriented.
- Every paragraph must add a new technical idea.
- Avoid repeating definitions already covered in earlier sections.
- Prefer concrete examples over abstract wording.
- Use short code comments only when they improve understanding.
- Do not write generic transition sentences like "In today's world" or "This is very important."
- Write as if explaining to a developer who may implement this after reading.
- When introducing a formula, explain what it means operationally.
- When describing a failure mode, mention how to detect or debug it.
"""



def worker_node(state: WorkerState) -> dict:
    task = Task(**state["task"])
    plan = Plan(**state["plan"])
    evidence = [EvidenceItem(**e) for e in state.get("evidence") or []]
    topic = state["topic"]
    mode = state.get("mode") or "closed_book"

    bullets_text = "\n- " + "\n- ".join(task.bullets)

    evidence_text = ""
    if evidence:
        evidence_text = "\n".join(
            f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}".strip()
            for e in evidence[:20]
        )

    section_md = llm_worker.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {topic}\n"
                    f"Mode: {mode}\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY use these URLs when citing):\n{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()

    return {"sections": [(task.id, section_md)]}


# -----------------------------
# 8) Reducer
# -----------------------------
class ImageSpec(BaseModel):
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")
    target_heading: str = Field(..., description="The EXACT heading text (without the ##) under which to place the image.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    images: List[ImageSpec] = Field(default_factory=list)

# ============================================================
# 8) ReducerWithImages (subgraph)
#    merge_content -> decide_images -> generate_and_place_images
# ============================================================
# ============================================================
#  FAST ReducerWithImages (SDXL-TURBO OPTIMIZED)
# ============================================================

from pathlib import Path
import os

# -------------------------------
# 1. Merge Content
# -------------------------------
def merge_content(state: State) -> dict:
    plan = state["plan"]

    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    
    # Strip LLM markdown wrapping if present
    if body.startswith("```markdown"):
        body = body[11:]
    if body.endswith("```"):
        body = body[:-3]
        
    merged_md = f"# {plan.blog_title}\n\n{body}\n"

    return {"merged_md": merged_md}


# -------------------------------
# 2. Decide Images (LIMITED TO 2)
# -------------------------------

DECIDE_IMAGES_SYSTEM = """You are an expert technical art director for a high-end developer blog.

Your job: plan exactly 2 striking hero images for the blog post.

STRICT RULES:
- Exactly 2 images. No more, no less.
- NO text, NO labels, NO arrows, NO diagrams, NO flowcharts in the image — AI image models cannot render readable text.
- Each image prompt must be 40-80 words, highly descriptive, specifying:
    * The core visual metaphor (e.g. "interconnected glowing neural nodes floating in dark space")
    * Art style: "photorealistic 3D render" OR "cinematic isometric illustration" OR "abstract neon concept art"
    * Color palette: e.g. "deep navy and electric blue with gold highlights"
    * Mood/lighting: e.g. "dramatic rim lighting, volumetric fog"
    * Quality boosters: "ultra-detailed, 8k resolution, professional studio quality"
- The `target_heading` must exactly match one of the ## section headings in the blog (copy it verbatim).
- First image: hero banner concept representing the overall topic.
- Second image: a mid-article visual representing a specific technical sub-concept.

Return strictly GlobalImagePlan JSON.
"""

def decide_images(state: State) -> dict:
    planner = llm_fast.with_structured_output(GlobalImagePlan)

    image_plan = planner.invoke([
        SystemMessage(content=DECIDE_IMAGES_SYSTEM),
        HumanMessage(content=state["merged_md"]),
    ])
    
    images = image_plan.images

    # 🔥 force at least 1 image
    if not images:
        images = [ImageSpec(
        filename="diagram.png",
        alt="diagram",
        caption="Generated diagram",
        prompt="simple technical diagram",
        target_heading="",
    )]

    return {
        "md_with_placeholders": state["merged_md"], # Legacy field, kept for state compatibility
        "image_specs": [img.model_dump() for img in images[:2]],  # force max 2
    }
   


# -------------------------------
#  BEST FREE IMAGE PIPELINE (FLUX + PRO TEXT OVERLAY)
# -------------------------------

from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import textwrap


def generate_flux_image(prompt: str):
    """
    Generate a high-quality image via Pollinations.ai using the flux-pro model.
    Falls back to flux if flux-pro fails.
    """
    # Enrich prompt for technical blog quality
    enriched_prompt = (
        f"{prompt}, "
        "ultra-detailed, professional technical illustration, "
        "dark background, vibrant accent colors, minimalist modern design, "
        "8k, sharp focus, no text, no watermark"
    )
    encoded_prompt = urllib.parse.quote(enriched_prompt)

    models_to_try = ["flux-pro", "flux"]
    last_exc = None

    for model in models_to_try:
        try:
            url = (
                f"https://image.pollinations.ai/prompt/{encoded_prompt}"
                f"?width=1280&height=720&model={model}&enhance=true&nologo=true&seed={hash(prompt) % 99999}"
            )
            response = requests.get(url, timeout=90)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGBA")
            print(f"Image generated with model: {model}")
            return img
        except Exception as e:
            last_exc = e
            print(f"Model {model} failed: {e}, trying next…")

    raise RuntimeError(f"All image models failed. Last error: {last_exc}")


def add_text_overlay(image: Image.Image, title: str, subtitle: str = ""):
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font_title = ImageFont.truetype("arial.ttf", 60)
        font_sub = ImageFont.truetype("arial.ttf", 35)
    except:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    W, H = image.size
    overlay_height = int(H * 0.35)

    draw.rectangle(
        [(0, H - overlay_height), (W, H)],
        fill=(0, 0, 0, 180)
    )

    draw.text((50, H - overlay_height + 40), title, font=font_title, fill=(255, 255, 255, 255))

    if subtitle:
        draw.text((50, H - overlay_height + 120), subtitle, font=font_sub, fill=(220, 220, 220, 255))

    combined = Image.alpha_composite(image, overlay)
    return combined.convert("RGB")


def _sd_generate_image(prompt: str, output_path: Path, title: str = "", subtitle: str = ""):
    try:
        img = generate_flux_image(prompt)

        # 🔥 SHORT TITLE for image (important improvement)
        short_title = " ".join(title.split()[:5])

        img = add_text_overlay(img, short_title, subtitle)
        img.save(output_path, quality=95)

    except Exception as e:
        print(f"Image generation failed: {e}")

import re

def sanitize_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = name.replace("–", "-").replace("—", "-").strip().strip(".")
    return name or "output"
    
# -------------------------------
# 4. Generate + Place Images
# -------------------------------

def generate_and_place_images(state: State) -> dict:

    plan = state["plan"]

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []

    # Skip images if none
    if not image_specs:
        filename = f"{plan.blog_title}.md"
        Path(filename).write_text(md, encoding="utf-8")
        return {"final": md}

    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    for spec in image_specs:
        filename = spec["filename"]
        out_path = images_dir / filename
        heading = spec.get("target_heading", "")

        try:
            # ⚡ Generate fast image via open API
            _sd_generate_image(spec["prompt"], out_path, title=plan.blog_title, subtitle=spec.get("alt", ""))

            img_md = f"""
<p align="center">
  <img src="images/{filename}" width="600"/>
</p>
<p align="center"><em>{spec['caption']}</em></p>
"""
            
            # Place after the target heading
            if heading and f"## {heading}" in md:
                md = md.replace(f"## {heading}", f"## {heading}\n\n{img_md}\n\n")
            else:
                # Fallback: put it at the very top under the H1
                md = re.sub(r"(# .*?\n)", rf"\1\n{img_md}\n\n", md, count=1)

        except Exception as e:
            # graceful fallback (no blocking)
            print(f"Image generation failed: {e}")

    filename = f"{plan.blog_title}.md"
    Path(filename).write_text(md, encoding="utf-8")
    return {"final": md}
    
# build reducer subgraph
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()

reducer_subgraph

# -----------------------------
# 9) Build graph
# -----------------------------
# -----------------------------
# 9) Build main graph
# -----------------------------
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout)
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()
app

# -----------------------------
# 10) Runner
# FIX: Always pass ALL required State keys to app.invoke()
# Missing keys cause KeyError inside nodes.
# -----------------------------
def run(topic: str):
    out = app.invoke(
        {
            "topic": topic,
            "mode": "",           # will be set by router_node
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
    )
    return out


# FIX: Last cell — use run() helper which initialises all required state keys.
if __name__ == "__main__":
    result=run("Self Attention in Transformer Architecture")
    print(result["final"])
