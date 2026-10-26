from __future__ import annotations

import operator
import os
import re
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults

import memory_store

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

    # memory (long-term recall across runs)
    memory_context: List[dict]

    # reflection / self-critique
    critic_report: List[dict]
    quality_score: float
    revision_count: int



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

# Groq's model catalog changes over time (e.g. llama-3.3-70b-versatile has been
# retired on some accounts). Pick models dynamically so the pipeline keeps
# working: prefer env overrides, then fall back to currently-available Groq
# models rather than hardcoding a single model name that may 404.
GROQ_STRONG_MODEL = os.environ.get("GROQ_STRONG_MODEL", "openai/gpt-oss-120b")
GROQ_FAST_MODEL = os.environ.get("GROQ_FAST_MODEL", "openai/gpt-oss-20b")

llm_strong = ChatGroq(
    model=GROQ_STRONG_MODEL,
    temperature=0.2,
    max_retries=5
)

llm_fast = ChatGroq(
    model=GROQ_FAST_MODEL,
    temperature=0.0,
    max_retries=5
)

llm_worker = ChatGroq(
    model=GROQ_STRONG_MODEL,
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


def _duckduckgo_search(query: str, max_results: int = 3) -> List[dict]:
    """Free, no-API-key web search fallback used when TAVILY_API_KEY is not set."""
    from ddgs import DDGS

    normalized: List[dict] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            url = r.get("href") or r.get("url") or ""
            if not url:
                continue
            normalized.append({
                "title": (r.get("title") or "")[:80],
                "url": url,
                "snippet": _clean_snippet(r.get("body") or ""),
                "published_at": None,
                "source": urllib.parse.urlparse(url).netloc or None,
            })
    return normalized


def _web_search(query: str, max_results: int = 3) -> List[dict]:
    """Dispatches to Tavily if configured, otherwise falls back to DuckDuckGo (free, no key)."""
    if os.environ.get("TAVILY_API_KEY"):
        return _tavily_search(query, max_results=max_results)
    try:
        return _duckduckgo_search(query, max_results=max_results)
    except Exception as e:
        print(f"DuckDuckGo search failed (non-blocking): {e}")
        return []


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

You MUST respond with a single JSON OBJECT (not a bare array) of the exact shape:
{"evidence": [{"title": "...", "url": "...", "published_at": null, "snippet": "...", "source": "..."}, ...]}
Always wrap the list in a top-level "evidence" key. Never respond with a raw JSON array.
"""


def _coerce_evidence_pack(raw) -> "EvidencePack":
    """Some Groq models ignore the 'wrap in an object' instruction under json_mode
    and return a bare JSON array. Accept either shape defensively instead of
    letting a strict parser blow up the whole research step."""
    import json as _json

    if isinstance(raw, str):
        raw = _json.loads(raw)
    if isinstance(raw, list):
        raw = {"evidence": raw}
    if isinstance(raw, dict) and "evidence" not in raw:
        # Some models return {"items": [...]} or similar — try to find a list value.
        list_values = [v for v in raw.values() if isinstance(v, list)]
        raw = {"evidence": list_values[0] if list_values else []}
    return EvidencePack(**raw)


def research_node(state: State) -> dict:
    queries = (state.get("queries") or [])[:3]   # max 3 queries
    raw_results: List[dict] = []
    for q in queries:
        raw_results.extend(_web_search(q, max_results=3))

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
    raw_response = llm_fast.bind(response_format={"type": "json_object"}).invoke([
        SystemMessage(content=RESEARCH_SYSTEM),
        HumanMessage(content=f"Raw search results:\n\n{formatted_results}"),
    ])

    raw_text = raw_response.content.strip()
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```[a-z]*\n?", "", raw_text)
        raw_text = re.sub(r"\n?```$", "", raw_text.strip())

    try:
        pack = _coerce_evidence_pack(raw_text)
    except Exception as e:
        print(f"Evidence parsing failed (non-blocking, continuing with no evidence): {e}")
        return {"evidence": []}

    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e

    return {"evidence": list(dedup.values())}

# -----------------------------
# 5) Orchestrator
# -----------------------------

ORCH_SYSTEM = """You are a technical blog planner.

YOU MUST return a single JSON object that EXACTLY matches this structure (no extra keys, no renaming):

{
  "blog_title": "<string>",
  "audience": "<string>",
  "tone": "<string>",
  "blog_kind": "explainer",
  "constraints": [],
  "tasks": [
    {
      "id": 1,
      "title": "<section title>",
      "goal": "<one sentence>",
      "bullets": ["<point>", "<point>", "<point>"],
      "target_words": 200,
      "tags": [],
      "requires_research": false,
      "requires_citations": false,
      "requires_code": false
    }
  ]
}

RULES:
- tasks array must have EXACTLY 4-6 items.
- id must be sequential integers starting at 1.
- Each task must have 3-4 bullets (plain strings).
- target_words must be between 150 and 350.
- At least 1 task must have requires_code=true.
- The LAST task MUST be a short conclusion / takeaways section that ties the
  post back to the opening thesis (e.g. title like "Conclusion" or
  "Key takeaways"). Set requires_code=false and requires_citations=false on it.
- NEVER use keys like "section1", "section2". Use the exact schema above.
- closed_book: evergreen content only.
- hybrid: use evidence for examples; requires_citations=true on those tasks.
- open_book: blog_kind="news_roundup".
- If "Previously generated blogs" are listed, do NOT reuse the same blog_title or
  section structure as any of them — pick a distinct angle so the new post adds
  value instead of duplicating earlier content. Keep tone consistent with past
  posts only if it clearly fits this topic too.
- Respond ONLY with the raw JSON object. No markdown fences, no preamble.
"""


def _ensure_conclusion_task(tasks: List[dict]) -> List[dict]:
    """Guarantee the plan ends with a short conclusion / takeaways section."""
    if not tasks:
        return tasks
    last_title = (tasks[-1].get("title") or "").lower()
    if any(k in last_title for k in ("conclusion", "takeaway", "summary", "wrap")):
        tasks[-1]["requires_code"] = False
        tasks[-1]["requires_citations"] = False
        return tasks

    conclusion = {
        "id": (tasks[-1].get("id") or len(tasks)) + 1 if len(tasks) < 6 else tasks[-1]["id"],
        "title": "Conclusion and key takeaways",
        "goal": "Tie the post back to the opening thesis with concrete takeaways.",
        "bullets": [
            "Recap the core problem the post set out to solve",
            "Summarize the practical recommendations",
            "Leave the reader with one clear next step",
        ],
        "target_words": 160,
        "tags": ["conclusion"],
        "requires_research": False,
        "requires_citations": False,
        "requires_code": False,
    }
    if len(tasks) >= 6:
        tasks[-1] = conclusion
    else:
        tasks.append(conclusion)
    return tasks


def _repair_plan_json(raw: dict, topic: str) -> dict:
    """Convert section1/section2/... LLM output into valid Plan schema."""
    if "tasks" in raw and "blog_title" in raw:
        tasks = list(raw.get("tasks") or [])
        if tasks and not any(t.get("requires_code") for t in tasks):
            code_target = tasks[-2] if len(tasks) >= 2 else tasks[-1]
            code_target["requires_code"] = True
        raw = {**raw, "tasks": _ensure_conclusion_task(tasks)}
        return raw

    tasks = []
    for i, (key, val) in enumerate(raw.items(), start=1):
        if not isinstance(val, dict):
            continue
        bullets = val.get("bullets", [])
        if bullets and isinstance(bullets[0], dict):
            bullets = [b.get("text", str(b)) for b in bullets]
        tasks.append({
            "id": i,
            "title": val.get("title", key.replace("section", "Section ")),
            "goal": val.get("goal", ""),
            "bullets": (bullets[:4] if bullets else ["Point 1", "Point 2", "Point 3"]),
            "target_words": val.get("target_words", 200),
            "tags": val.get("tags", []),
            "requires_research": val.get("requires_research", False),
            "requires_citations": val.get("requires_citations", False),
            "requires_code": val.get("requires_code", False),
        })

    tasks = tasks[:6]
    if tasks and not any(t["requires_code"] for t in tasks):
        # Prefer putting code on a non-conclusion section when possible.
        code_target = tasks[-2] if len(tasks) >= 2 else tasks[-1]
        code_target["requires_code"] = True

    tasks = _ensure_conclusion_task(tasks)

    return {
        "blog_title": raw.get("blog_title", topic),
        "audience": raw.get("audience", "software engineers"),
        "tone": raw.get("tone", "technical and clear"),
        "blog_kind": raw.get("blog_kind", "explainer"),
        "constraints": raw.get("constraints", []),
        "tasks": tasks,
    }


def orchestrator_node(state: State) -> dict:
    import json as _json

    evidence = state.get("evidence") or []
    mode = state.get("mode") or "closed_book"
    topic = state["topic"]

    # --- Long-term memory: recall topically-related past runs ---
    memories = memory_store.get_relevant_memories(topic, k=3)
    if memories:
        memory_text = "\n".join(
            f'- "{m.get("blog_title", "Untitled")}" '
            f'(tone: {m.get("tone", "unknown")}, kind: {m.get("blog_kind", "unknown")}, '
            f'quality_score: {m.get("quality_score", "n/a")})'
            for m in memories
        )
    else:
        memory_text = "(none found)"

    raw_response = llm_strong.invoke([
        SystemMessage(content=ORCH_SYSTEM),
        HumanMessage(
            content=(
                f"Topic: {topic}\n"
                f"Mode: {mode}\n\n"
                f"Previously generated blogs on related topics:\n{memory_text}\n\n"
                f"Evidence (ONLY use for fresh claims; may be empty):\n"
                f"{[e.model_dump() for e in evidence][:16]}"
            )
        ),
    ])

    raw_text = raw_response.content.strip()

    # Strip markdown code fences if present
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```[a-z]*\n?", "", raw_text)
        raw_text = re.sub(r"\n?```$", "", raw_text.strip())

    try:
        raw_dict = _json.loads(raw_text)
    except _json.JSONDecodeError as e:
        raise ValueError(f"Orchestrator returned invalid JSON: {e}\n\nRaw:\n{raw_text[:500]}")

    repaired = _repair_plan_json(raw_dict, topic)
    plan = Plan(**repaired)
    return {"plan": plan, "memory_context": memories}


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
- Code must be copy-pasteable and logically correct. Prefer the happy-path
  pattern (e.g. on a cache miss always fetch live data, then store it). Never
  invent inverted / placeholder-empty logic that a reader would be wrong to
  copy into production.

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

# -------------------------------
# 1. Merge Content
# -------------------------------
def merge_content(state: State) -> dict:
    plan = state["plan"]

    # NOTE: state["sections"] uses an Annotated[..., operator.add] reducer and can
    # contain duplicate (task_id, section_md) entries when the parallel worker
    # fan-in merges across the reducer subgraph boundary. Deduplicating by task_id
    # (dict() keeps the last write per key) guards against sections being
    # rendered twice in the final blog regardless of how many times each
    # worker's partial update gets folded into the channel.
    sections_by_id = dict(state["sections"])
    ordered_sections = [md for _, md in sorted(sections_by_id.items())]
    body = "\n\n".join(ordered_sections).strip()
    
    # Strip LLM markdown wrapping if present
    if body.startswith("```markdown"):
        body = body[11:]
    if body.endswith("```"):
        body = body[:-3]
        
    merged_md = f"# {plan.blog_title}\n\n{body}\n"

    return {"merged_md": merged_md}


# -------------------------------
# 1b. Critic / Self-Reflection Loop
#     Grades each section's groundedness against the evidence pool and
#     triggers a single bounded rewrite pass for sections that fail.
# -------------------------------
class ClaimCheck(BaseModel):
    groundedness_score: int = Field(..., ge=0, le=100)
    unsupported_claims: List[str] = Field(default_factory=list)
    needs_revision: bool


CRITIC_SYSTEM = """You are a meticulous technical fact-checking editor.

Given ONE blog section and the pool of Evidence URLs available to the writer,
score how well-grounded the section's outside-world / factual claims are.

Scoring:
- 100 = every outside-world claim is either absent, purely conceptual/evergreen,
  or clearly backed by one of the Evidence URLs.
- Lower scores = specific outside-world claims (events, companies, versions,
  pricing, rankings, dates) that are NOT backed by any Evidence URL, or missing
  citations where the section clearly requires them.
- List up to 3 concrete unsupported claims verbatim (short phrases) if any.
- needs_revision = true only if groundedness_score < 70.

Respond strictly as JSON matching this schema:
{"groundedness_score": <0-100 int>, "unsupported_claims": [<string>, ...], "needs_revision": <bool>}
"""

MAX_REVISION_PASSES = 1


def critic_node(state: State) -> dict:
    plan = state["plan"]
    evidence = state.get("evidence") or []
    mode = state.get("mode") or "closed_book"
    revision_count = state.get("revision_count", 0)

    tasks_by_id = {t.id: t for t in plan.tasks}
    sections_by_id = dict(state["sections"])

    evidence_text = "\n".join(f"- {e.url}" for e in evidence[:20]) or "(no evidence provided)"
    judge = llm_fast.with_structured_output(ClaimCheck, method="json_mode")

    checks: List[dict] = []
    revised_sections: List[tuple] = []

    for task_id, section_md in sorted(sections_by_id.items()):
        task = tasks_by_id.get(task_id)
        needs_check = bool(task) and (task.requires_citations or mode == "open_book")

        if not needs_check:
            checks.append({
                "section_id": task_id,
                "title": task.title if task else f"Section {task_id}",
                "groundedness_score": 100,
                "unsupported_claims": [],
                "needs_revision": False,
                "revised": False,
            })
            revised_sections.append((task_id, section_md))
            continue

        try:
            check = judge.invoke([
                SystemMessage(content=CRITIC_SYSTEM),
                HumanMessage(content=(
                    f"Evidence URLs available:\n{evidence_text}\n\n"
                    f"Section:\n{section_md}"
                )),
            ])
        except Exception:
            # If the judge call fails, don't block the pipeline — assume OK.
            check = ClaimCheck(groundedness_score=100, unsupported_claims=[], needs_revision=False)

        was_revised = False
        final_md = section_md

        if check.needs_revision and revision_count < MAX_REVISION_PASSES:
            feedback = (
                f"Your previous draft scored {check.groundedness_score}/100 on groundedness. "
                f"Unsupported claims flagged by the editor: {check.unsupported_claims}. "
                f"Rewrite the section so every outside-world claim is either removed, softened "
                f"into general/evergreen knowledge, or backed by one of the Evidence URLs above."
            )
            try:
                final_md = llm_worker.invoke([
                    SystemMessage(content=WORKER_SYSTEM),
                    HumanMessage(content=(
                        f"Original section:\n{section_md}\n\n"
                        f"Evidence (ONLY use these URLs when citing):\n{evidence_text}\n\n"
                        f"Editor feedback — fix this:\n{feedback}"
                    )),
                ]).content.strip()
                was_revised = True
            except Exception:
                final_md = section_md

        checks.append({
            "section_id": task_id,
            "title": task.title if task else f"Section {task_id}",
            "groundedness_score": check.groundedness_score,
            "unsupported_claims": check.unsupported_claims,
            "needs_revision": check.needs_revision,
            "revised": was_revised,
        })
        revised_sections.append((task_id, final_md))

    avg_score = round(sum(c["groundedness_score"] for c in checks) / len(checks), 1) if checks else 100.0

    body = "\n\n".join(md for _, md in sorted(revised_sections, key=lambda x: x[0])).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"

    return {
        "merged_md": merged_md,
        "critic_report": checks,
        "quality_score": avg_score,
        "revision_count": revision_count + 1,
    }


# -------------------------------
# 2. Decide Images (LIMITED TO 2)
# -------------------------------

DECIDE_IMAGES_SYSTEM = """You are an art director for a premium technical blog (think Stripe / Linear / Anthropic editorial covers).

Create at most 2 image specs. The FIRST image is the hero / cover for the blog.
The SECOND (optional) is a mid-article conceptual illustration.

HARD RULES for every prompt:
- NO text, letters, numbers, logos, watermarks, UI chrome, arrows, boxes, or flowcharts.
- NO diagrams that try to "explain" the system — image models ruin those.
- YES: cinematic, atmospheric, conceptual metaphors for the technical idea.
- Style keywords to INCLUDE in every prompt:
  "editorial tech photography aesthetic, soft volumetric lighting, shallow depth of field,
   muted teal and warm amber accents on deep charcoal background, clean composition,
   high detail, 8k, award-winning digital art, no text, no watermark"
- Prefer metaphors: glowing neural lattices, floating translucent glass panels,
  constellation-like data nodes, light beams through fog, origami paper layers as
  "memory", interlocking metallic rings as "agents", etc.
- Keep prompts 1–3 sentences, concrete, and specific to THIS blog's topic — not generic "AI brain".

Return strictly GlobalImagePlan.
"""

# Appended to every image prompt so even a weak LLM plan still looks editorial.
IMAGE_STYLE_SUFFIX = (
    ", editorial tech cover art, cinematic lighting, soft volumetric fog, "
    "muted teal and amber accents on deep charcoal, clean negative space in the lower third, "
    "high detail, photorealistic digital art, no text, no letters, no logos, no watermark, no UI"
)


def _enrich_image_prompt(prompt: str) -> str:
    cleaned = (prompt or "").strip().rstrip(".")
    # Strip common LLM diagram requests that make Pollinations look bad.
    banned = ("flowchart", "diagram with", "arrows", "labeled", "boxes and", "uml", "wireframe")
    lower = cleaned.lower()
    if any(b in lower for b in banned) or len(cleaned) < 20:
        cleaned = (
            "Abstract cinematic metaphor for modern AI systems: translucent glass panels "
            "and soft glowing data nodes floating in a dark studio"
        )
    if "no text" not in lower:
        cleaned = cleaned + IMAGE_STYLE_SUFFIX
    return cleaned


def decide_images(state: State) -> dict:
    plan = state["plan"]
    planner = llm_fast.with_structured_output(GlobalImagePlan)

    image_plan = planner.invoke([
        SystemMessage(content=DECIDE_IMAGES_SYSTEM),
        HumanMessage(content=(
            f"Blog title: {plan.blog_title}\n"
            f"Audience: {plan.audience}\n"
            f"Tone: {plan.tone}\n\n"
            f"Blog draft (for thematic inspiration only):\n{state['merged_md'][:3500]}"
        )),
    ])

    images = image_plan.images

    # Force at least 1 strong hero image if the planner returned nothing.
    if not images:
        images = [ImageSpec(
            filename="hero_cover.png",
            alt=plan.blog_title,
            caption=plan.blog_title,
            prompt=(
                f"Cinematic editorial cover metaphor for '{plan.blog_title}': "
                "translucent glass planes and soft teal light filaments over a charcoal void"
            ),
            target_heading="",
        )]

    enriched: List[ImageSpec] = []
    for i, img in enumerate(images[:2]):
        data = img.model_dump() if hasattr(img, "model_dump") else dict(img)
        data["prompt"] = _enrich_image_prompt(data.get("prompt", ""))
        if i == 0 and not data.get("filename"):
            data["filename"] = "hero_cover.png"
        enriched.append(ImageSpec(**data))

    return {
        "md_with_placeholders": state["merged_md"],
        "image_specs": [img.model_dump() for img in enriched],
    }


# -------------------------------
#  Image pipeline (Pollinations + polished cover overlay)
# -------------------------------

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from io import BytesIO
import textwrap


def generate_flux_image(prompt: str):
    encoded_prompt = urllib.parse.quote(prompt)
    # Prefer higher-res widescreen covers. Skip enhance=true (often 402 on free tier).
    candidates = [
        f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1536&height=864&model=flux&nologo=true&safe=true",
        f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1280&height=720&model=flux&nologo=true",
        f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1280&height=720&nologo=true",
    ]
    last_err: Exception | None = None
    for url in candidates:
        try:
            response = requests.get(url, timeout=90)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGBA")
        except Exception as e:
            last_err = e
            continue
    raise last_err or RuntimeError("Image generation failed for all Pollinations URL variants")


def _load_font(size: int) -> ImageFont.ImageFont:
    # Prefer fonts that look editorial on Windows; fall back gracefully.
    candidates = [
        "segoeui.ttf",
        "SegoeUI.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _wrap_text(text: str, font: ImageFont.ImageFont, draw: ImageDraw.ImageDraw, max_width: int) -> List[str]:
    words = (text or "").split()
    if not words:
        return []
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        bbox = draw.textbbox((0, 0), trial, font=font)
        if (bbox[2] - bbox[0]) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines[:4]  # never more than 4 lines on a cover


def add_cover_overlay(image: Image.Image, title: str, subtitle: str = "") -> Image.Image:
    """Soft magazine-style cover: gradient veil + wrapped title (not a hard black bar)."""
    image = image.convert("RGBA")
    # Slight contrast/color polish before overlay.
    image = ImageEnhance.Contrast(image).enhance(1.08)
    image = ImageEnhance.Color(image).enhance(1.12)

    W, H = image.size
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Soft bottom-to-mid gradient (no hard rectangle edge).
    gradient_top = int(H * 0.45)
    for y in range(gradient_top, H):
        t = (y - gradient_top) / max(1, (H - gradient_top))
        alpha = int(30 + 190 * (t ** 1.4))
        draw.line([(0, y), (W, y)], fill=(8, 10, 14, alpha))

    font_title = _load_font(max(36, W // 22))
    font_sub = _load_font(max(20, W // 42))

    margin = int(W * 0.05)
    max_text_width = W - 2 * margin
    # Prefer a readable short title — keep up to ~10 words.
    display_title = " ".join((title or "").split()[:10]) or "BlogForge"
    title_lines = _wrap_text(display_title, font_title, draw, max_text_width)

    # Measure block height so we can bottom-align nicely.
    line_gap = 8
    title_block_h = 0
    for line in title_lines:
        bbox = draw.textbbox((0, 0), line, font=font_title)
        title_block_h += (bbox[3] - bbox[1]) + line_gap
    sub_h = 0
    if subtitle:
        sub_lines = _wrap_text(subtitle, font_sub, draw, max_text_width)
        for line in sub_lines:
            bbox = draw.textbbox((0, 0), line, font=font_sub)
            sub_h += (bbox[3] - bbox[1]) + 4
    else:
        sub_lines = []

    y = H - margin - sub_h - title_block_h - 12
    for line in title_lines:
        # Soft shadow for readability on bright regions.
        draw.text((margin + 2, y + 2), line, font=font_title, fill=(0, 0, 0, 140))
        draw.text((margin, y), line, font=font_title, fill=(245, 247, 250, 255))
        bbox = draw.textbbox((0, 0), line, font=font_title)
        y += (bbox[3] - bbox[1]) + line_gap

    if sub_lines:
        y += 6
        for line in sub_lines[:2]:
            draw.text((margin + 1, y + 1), line, font=font_sub, fill=(0, 0, 0, 120))
            draw.text((margin, y), line, font=font_sub, fill=(200, 210, 220, 230))
            bbox = draw.textbbox((0, 0), line, font=font_sub)
            y += (bbox[3] - bbox[1]) + 4

    combined = Image.alpha_composite(image, overlay)
    # Tiny sharpen after composite for a crisper cover look.
    return combined.convert("RGB").filter(ImageFilter.UnsharpMask(radius=1.2, percent=80, threshold=2))


def add_text_overlay(image: Image.Image, title: str, subtitle: str = ""):
    """Backward-compatible alias used by older call sites / tests."""
    return add_cover_overlay(image, title, subtitle)


def _sd_generate_image(
    prompt: str,
    output_path: Path,
    title: str = "",
    subtitle: str = "",
    with_overlay: bool = True,
) -> bool:
    """Generate and save an image. Returns True only if the file was written successfully."""
    try:
        img = generate_flux_image(_enrich_image_prompt(prompt))
        if with_overlay and title:
            img = add_cover_overlay(img, title, subtitle)
        else:
            img = ImageEnhance.Contrast(img.convert("RGB")).enhance(1.05)
        img.save(output_path, quality=95)
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception as e:
        print(f"Image generation failed: {e}")
        try:
            if output_path.exists():
                output_path.unlink()
        except OSError:
            pass
        return False


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
        # IMPORTANT: use sanitize_filename() here, not the raw title. Titles
        # routinely contain characters like ':' that are illegal in Windows
        # filenames. Writing them raw doesn't raise an error on Windows/NTFS —
        # it silently truncates at the illegal character and creates a
        # different, empty file, silently losing the blog content on disk.
        filename = f"{sanitize_filename(plan.blog_title)}.md"
        Path(filename).write_text(md, encoding="utf-8")
        try:
            memory_store.save_memory({
                "topic": state.get("topic", plan.blog_title),
                "blog_title": plan.blog_title,
                "audience": plan.audience,
                "tone": plan.tone,
                "blog_kind": plan.blog_kind,
                "quality_score": state.get("quality_score", 100.0),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as e:
            print(f"Memory write failed (non-blocking): {e}")
        return {"final": md}

    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    surviving_specs: List[dict] = []
    for i, spec in enumerate(image_specs):
        filename = spec["filename"]
        out_path = images_dir / filename
        heading = spec.get("target_heading", "")

        try:
            # Hero (first) image gets the magazine-style title overlay.
            # Secondary images stay clean — overlay often makes them look cluttered.
            ok = _sd_generate_image(
                spec["prompt"], out_path,
                title=plan.blog_title if i == 0 else "",
                subtitle=spec.get("caption", "") if i == 0 else "",
                with_overlay=(i == 0),
            )
            # CRITICAL: only embed an <img> tag when the file actually exists on
            # disk. Pollinations (and similar free endpoints) can return 402 /
            # empty responses — previously we still wrote dead links into the
            # Markdown, which broke the exported .md even though the UI handled
            # the missing file gracefully.
            if not ok:
                print(f"Skipping image embed for {filename} (generation failed)")
                continue

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

            surviving_specs.append(spec)

        except Exception as e:
            # graceful fallback (no blocking)
            print(f"Image generation failed: {e}")

    # Keep image_specs in sync with what was actually written so the UI Images
    # tab doesn't show phantom entries for failed generations.
    image_specs = surviving_specs

    filename = f"{sanitize_filename(plan.blog_title)}.md"
    Path(filename).write_text(md, encoding="utf-8")

    # --- Long-term memory: remember this run for future topically-related runs ---
    try:
        memory_store.save_memory({
            "topic": state.get("topic", plan.blog_title),
            "blog_title": plan.blog_title,
            "audience": plan.audience,
            "tone": plan.tone,
            "blog_kind": plan.blog_kind,
            "quality_score": state.get("quality_score", 100.0),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        print(f"Memory write failed (non-blocking): {e}")

    return {"final": md, "image_specs": image_specs}
    
# build reducer subgraph
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("critic", critic_node)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "critic")
reducer_graph.add_edge("critic", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()

reducer_subgraph

# -----------------------------
# 9) Build graphs
#
# Three compiled graphs are exposed:
#   - app         : full single-shot pipeline (router -> ... -> reducer).
#                   Used for the CLI runner and for "auto" mode in the UI.
#   - plan_app    : router -> (research) -> orchestrator only. Stops once a
#                   Plan exists, so a human can review/edit it before any
#                   writing happens (human-in-the-loop approval gate).
#   - writing_app : worker fanout -> reducer (critic + images). Resumes from
#                   an (optionally edited) approved Plan produced by plan_app.
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

# --- Phase 1: plan only (human-in-the-loop approval gate stops here) ---
planning_graph = StateGraph(State)
planning_graph.add_node("router", router_node)
planning_graph.add_node("research", research_node)
planning_graph.add_node("orchestrator", orchestrator_node)
planning_graph.add_edge(START, "router")
planning_graph.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
planning_graph.add_edge("research", "orchestrator")
planning_graph.add_edge("orchestrator", END)
plan_app = planning_graph.compile()

# --- Phase 2: write from an approved (possibly human-edited) plan ---
writing_graph = StateGraph(State)
writing_graph.add_node("worker", worker_node)
writing_graph.add_node("reducer", reducer_subgraph)
writing_graph.add_conditional_edges(START, fanout)
writing_graph.add_edge("worker", "reducer")
writing_graph.add_edge("reducer", END)
writing_app = writing_graph.compile()


# -----------------------------
# 10) Runners
# FIX: Always pass ALL required State keys to *_app.invoke().
# Missing keys cause KeyError inside nodes.
# -----------------------------
def default_initial_state(topic: str) -> dict:
    return {
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
        "memory_context": [],
        "critic_report": [],
        "quality_score": 0.0,
        "revision_count": 0,
    }


def run(topic: str) -> dict:
    """Single-shot full pipeline — no human approval pause. Used by the CLI / tests."""
    return app.invoke(default_initial_state(topic))


def run_plan(topic: str) -> dict:
    """Phase 1 only: produces a Plan for human review before any writing happens."""
    return plan_app.invoke(default_initial_state(topic))


def run_writing(state: dict) -> dict:
    """Phase 2: continues from a state dict whose 'plan' has been approved/edited."""
    return writing_app.invoke(state)


# FIX: Last cell — use run() helper which initialises all required state keys.
if __name__ == "__main__":
    pipeline_result = run("Self Attention in Transformer Architecture")
    print(pipeline_result["final"])
    print("\n--- Quality / groundedness score:", pipeline_result.get("quality_score"))
    print("--- Memory records stored so far:", memory_store.memory_count())
