"""
Offline end-to-end test for the BlogForge AI pipeline.

This monkeypatches every external call (Groq LLM calls, Tavily search, and
the Pollinations image API) with deterministic fakes so the *entire* LangGraph
wiring — router -> research -> memory recall -> orchestrator -> human-in-the-
loop plan/writing split -> parallel worker fanout -> critic/self-correction
loop -> image placement -> memory persistence — can be exercised without any
real API keys, network access, or cost.

Run with:  .venv\\Scripts\\python.exe test_pipeline_offline.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Dummy keys so module-level ChatGroq(...) construction doesn't fail on import.
os.environ.setdefault("GROQ_API_KEY", "test-key-not-real")
os.environ.setdefault("TAVILY_API_KEY", "test-key-not-real")

# Use an isolated memory file for this test run so it doesn't pollute real history.
TEST_MEMORY_FILE = Path("memory_store.test.json")
if TEST_MEMORY_FILE.exists():
    TEST_MEMORY_FILE.unlink()

import memory_store  # noqa: E402
memory_store.MEMORY_FILE = TEST_MEMORY_FILE

import bwa_research_image as bwa  # noqa: E402


PASS = []
FAIL = []


def check(name: str, condition: bool, detail: str = ""):
    if condition:
        PASS.append(name)
        print(f"  [PASS] {name}")
    else:
        FAIL.append(f"{name} :: {detail}")
        print(f"  [FAIL] {name} :: {detail}")


# ---------------------------------------------------------------------------
# Fake LLM structured-output responses, keyed by the Pydantic schema requested
# ---------------------------------------------------------------------------
def fake_router_decision(*args, **kwargs):
    return bwa.RouterDecision(
        needs_research=True,
        mode="hybrid",
        queries=["self attention transformers", "attention mechanism explained"],
    )


def fake_evidence_pack_json(*args, **kwargs):
    """research_node now calls llm_fast.bind(...).invoke(...) and parses raw JSON text
    itself (to defensively handle models that return a bare array instead of
    {"evidence": [...]}). Return a message-like object with .content JSON text."""
    msg = MagicMock()
    msg.content = json.dumps({
        "evidence": [{
            "title": "Attention Is All You Need",
            "url": "https://arxiv.org/abs/1706.03762",
            "published_at": "2017-06-12",
            "snippet": "Introduces the Transformer architecture.",
            "source": "arxiv.org",
        }]
    })
    return msg


def fake_plan_response(*args, **kwargs):
    plan_dict = {
        "blog_title": "Understanding Self-Attention in Transformers",
        "audience": "ML engineers",
        "tone": "technical and clear",
        "blog_kind": "explainer",
        "constraints": [],
        "tasks": [
            {
                "id": 1, "title": "What Self-Attention Solves", "goal": "Explain the motivation.",
                "bullets": ["Sequence modeling limits of RNNs", "Parallelism", "Long-range dependencies"],
                "target_words": 180, "tags": [], "requires_research": False,
                "requires_citations": True, "requires_code": False,
            },
            {
                "id": 2, "title": "The QKV Mechanism", "goal": "Explain query/key/value math.",
                "bullets": ["Query/Key/Value projections", "Scaled dot product", "Softmax weighting"],
                "target_words": 220, "tags": [], "requires_research": False,
                "requires_citations": False, "requires_code": False,
            },
            {
                "id": 3, "title": "Multi-Head Attention", "goal": "Explain multiple heads.",
                "bullets": ["Why multiple heads", "Concatenation", "Learned subspaces"],
                "target_words": 200, "tags": [], "requires_research": False,
                "requires_citations": False, "requires_code": False,
            },
            {
                "id": 4, "title": "Minimal PyTorch Implementation", "goal": "Show working code.",
                "bullets": ["Linear projections", "Attention scores", "Weighted sum output"],
                "target_words": 220, "tags": [], "requires_research": False,
                "requires_citations": False, "requires_code": True,
            },
            {
                "id": 5, "title": "Conclusion and key takeaways", "goal": "Tie the post back to the thesis.",
                "bullets": ["Recap the problem", "Summarize recommendations", "One clear next step"],
                "target_words": 160, "tags": ["conclusion"], "requires_research": False,
                "requires_citations": False, "requires_code": False,
            },
        ],
    }
    msg = MagicMock()
    msg.content = json.dumps(plan_dict)
    return msg


def fake_claim_check_low_score(*args, **kwargs):
    return bwa.ClaimCheck(
        groundedness_score=55,
        unsupported_claims=["Transformers fully replaced RNNs in production by 2018"],
        needs_revision=True,
    )


def fake_image_plan(*args, **kwargs):
    return bwa.GlobalImagePlan(images=[
        bwa.ImageSpec(
            filename="qkv_flow.png", alt="qkv diagram", caption="QKV attention flow",
            prompt="minimalist isometric vector art of attention flow", target_heading="The QKV Mechanism",
        ),
    ])


class FakeLLM:
    """Stands in for llm_strong / llm_fast / llm_worker depending on call site."""

    def __init__(self, kind: str):
        self.kind = kind  # "strong" | "fast_router" | "fast_evidence" | "fast_critic" | "fast_imageplan" | "worker"

    def bind(self, **kwargs):
        """research_node calls llm_fast.bind(response_format=...).invoke(...) and
        parses the raw JSON text itself. Return a proxy whose invoke() returns
        a fake evidence-pack JSON response regardless of bind kwargs."""
        proxy = MagicMock()
        proxy.invoke.side_effect = fake_evidence_pack_json
        return proxy

    def with_structured_output(self, schema, method=None):
        proxy = MagicMock()
        if schema is bwa.RouterDecision:
            proxy.invoke.side_effect = fake_router_decision
        elif schema is bwa.ClaimCheck:
            proxy.invoke.side_effect = fake_claim_check_low_score
        elif schema is bwa.GlobalImagePlan:
            proxy.invoke.side_effect = fake_image_plan
        else:
            raise AssertionError(f"Unexpected schema requested: {schema}")
        return proxy

    def invoke(self, messages, *args, **kwargs):
        # Only llm_strong.invoke (orchestrator) and llm_worker.invoke (sections) call this directly.
        system = messages[0].content if messages else ""
        if "technical blog planner" in system:
            return fake_plan_response()
        if "senior technical writer" in system:
            msg = MagicMock()
            msg.content = (
                "## Fake Section\n\nThis is deterministic fake section content used for "
                "offline testing. It mentions ([arxiv.org](https://arxiv.org/abs/1706.03762)) "
                "as a citation when evidence is relevant.\n"
            )
            return msg
        raise AssertionError(f"Unexpected system prompt routed to FakeLLM.invoke: {system[:80]}")


def fake_image_get(url, timeout=60):
    """Stand in for requests.get() used by generate_flux_image — returns a tiny valid PNG."""
    from PIL import Image
    from io import BytesIO
    buf = BytesIO()
    Image.new("RGB", (64, 64), color=(10, 10, 10)).save(buf, format="PNG")
    resp = MagicMock()
    resp.content = buf.getvalue()
    resp.raise_for_status = lambda: None
    return resp


def main():
    print("=" * 70)
    print("BlogForge AI — OFFLINE end-to-end pipeline test (mocked LLM/search/image)")
    print("=" * 70)

    with patch.object(bwa, "llm_strong", FakeLLM("strong")), \
         patch.object(bwa, "llm_fast", FakeLLM("fast")), \
         patch.object(bwa, "llm_worker", FakeLLM("worker")), \
         patch.object(bwa, "_tavily_search", lambda query, max_results=3: [
             {"title": "Attention Is All You Need", "url": "https://arxiv.org/abs/1706.03762",
              "snippet": "Transformer paper", "published_at": "2017-06-12", "source": "arxiv.org"}
         ]), \
         patch("requests.get", side_effect=fake_image_get):

        topic = "Self Attention in Transformer Architecture"

        # ---- Stage 1: Plan (human-in-the-loop checkpoint) ----
        print("\n[1] Running plan_app (router -> research -> memory recall -> orchestrator)...")
        plan_state = bwa.run_plan(topic)

        check("router set needs_research", plan_state.get("needs_research") is True)
        check("router set mode=hybrid", plan_state.get("mode") == "hybrid")
        check("research produced evidence", len(plan_state.get("evidence", [])) == 1)
        check("orchestrator produced a Plan", plan_state.get("plan") is not None)
        check("plan has 5 tasks (incl. conclusion)", len(plan_state["plan"].tasks) == 5)
        check(
            "last plan task is a conclusion",
            "conclusion" in plan_state["plan"].tasks[-1].title.lower(),
        )
        check("memory_context key present (empty on first run)", plan_state.get("memory_context") == [])

        # ---- Simulate human-in-the-loop edit: rename section 1 ----
        print("\n[2] Simulating a human edit to the plan before approval...")
        edited_plan = plan_state["plan"].model_dump()
        edited_plan["tasks"][0]["title"] = "What Self-Attention Solves (edited by human)"
        plan_state["plan"] = bwa.Plan(**edited_plan)
        check("human edit applied to plan", "(edited by human)" in plan_state["plan"].tasks[0].title)

        # ---- Stage 2: Writing (fanout workers -> critic/self-correction -> images) ----
        print("\n[3] Running writing_app (fanout workers -> critic -> images)...")
        final_state = bwa.run_writing(plan_state)

        unique_section_ids = {tid for tid, _ in final_state.get("sections", [])}
        check("exactly 5 unique sections were written (one per task)", unique_section_ids == {1, 2, 3, 4, 5},
              f"got unique ids={unique_section_ids}")
        check("final markdown produced", bool(final_state.get("final")))
        check(
            "no duplicate section headings in final blog "
            "(regression check: LangGraph fan-in can duplicate the 'sections' "
            "channel across the reducer subgraph boundary; merge_content must dedupe by id)",
            final_state["final"].count("## Fake Section") == 5,
            f"found {final_state['final'].count('## Fake Section')} occurrences, expected 5",
        )
        check("human-edited plan was carried through to the writing phase unchanged",
              "(edited by human)" in final_state["plan"].tasks[0].title)

        # ---- Critic / reflection checks ----
        print("\n[4] Checking critic/self-correction report...")
        critic_report = final_state.get("critic_report", [])
        check("critic_report has one entry per section", len(critic_report) == 5)
        citation_checks = [c for c in critic_report if c["section_id"] == 1]
        check("section 1 (requires_citations) was actually graded", len(citation_checks) == 1)
        if citation_checks:
            check("low-scoring section flagged needs_revision", citation_checks[0]["needs_revision"] is True)
            check("low-scoring section was revised (self-correction fired)", citation_checks[0]["revised"] is True)
        non_citation_checks = [c for c in critic_report if c["section_id"] != 1]
        check("non-citation sections auto-scored 100 (skipped judge call)",
              all(c["groundedness_score"] == 100 and not c["revised"] for c in non_citation_checks))
        check("quality_score computed", isinstance(final_state.get("quality_score"), float))

        # ---- Image pipeline ----
        print("\n[5] Checking image generation + placement...")
        image_specs = final_state.get("image_specs", [])
        check("exactly 1 image planned", len(image_specs) == 1)
        img_path = Path("images") / image_specs[0]["filename"]
        check("image file was written to disk", img_path.exists())
        check("image embedded into final markdown", "qkv_flow.png" in final_state["final"])

        # ---- Memory persistence ----
        print("\n[6] Checking long-term memory persistence...")
        check("memory file created", TEST_MEMORY_FILE.exists())
        stored = memory_store.all_memories()
        check("one memory record stored", len(stored) == 1)
        check("stored record has blog_title", stored[0].get("blog_title") == final_state["plan"].blog_title)

        # ---- Memory recall on a second, related run ----
        print("\n[7] Running a second related topic to verify memory recall...")
        plan_state_2 = bwa.run_plan("Multi-Head Self Attention in Transformer Models")
        recalled = plan_state_2.get("memory_context", [])
        check("second run recalled the first blog from memory", len(recalled) >= 1,
              f"recalled={recalled}")

    print("\n" + "=" * 70)
    print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("\nFailures:")
        for f in FAIL:
            print(f"  - {f}")
    print("=" * 70)

    # cleanup test artifacts
    if TEST_MEMORY_FILE.exists():
        TEST_MEMORY_FILE.unlink()

    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
