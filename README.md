# ✦ BlogForge AI — Multi-Agent Content Generation System

> An end-to-end AI pipeline that researches, plans, writes, and illustrates technical blog posts — fully automated using LangGraph, Groq (LLaMA 3.3), Tavily, and a Streamlit UI.

---

##  Overview

BlogForge AI takes a single topic as input and produces a fully-formatted, research-backed Markdown blog post — complete with AI-generated cover images — in 1–2 minutes.

The system is built on a **multi-agent LangGraph pipeline** where specialised agents handle routing, web research, content planning, parallel section writing, and image generation — each with clearly scoped responsibilities.

---

##  Architecture

```mermaid
flowchart TD
    A([🗂️ Topic Input]) --> B

    subgraph MAIN["🔁 LangGraph Main Graph"]
        B["🔀 Router Agent\nDecides research mode\nclosed_book / hybrid / open_book"]

        B -->|needs_research = false| D
        B -->|needs_research = true| C

        subgraph RESEARCH["🌐 Research Phase"]
            C["🔎 Research Agent\nTavily web search\nLLM deduplication & validation"]
        end

        C -->|EvidencePack| D

        D["🧠 Orchestrator Agent\nGenerates structured Plan\n4–6 sections with goals,\nbullets & word targets"]

        D -->|"Send() fan-out"| E

        subgraph WORKERS["⚡ Parallel Workers  — LangGraph Send()"]
            E1["✍️ Worker 1\nSection"]
            E2["✍️ Worker 2\nSection"]
            E3["✍️ Worker N\nSection"]
        end

        E --> E1 & E2 & E3

        E1 & E2 & E3 -->|sections accumulate| F

        subgraph REDUCER["📦 Reducer Subgraph"]
            F["🔗 Merge Content\nOrdered assembly of sections"]
            F --> G["🖼️ Decide Images\nLLM picks ≤2 image placements"]
            G --> H["🎨 Generate & Place Images\nFlux via Pollinations API\nPillow text overlay\nEmbed into Markdown"]
        end
    end

    H --> I([📄 Final Markdown\n+ images/ folder])
    I --> J["🖥️ Streamlit UI\nBlogForge AI"]

    subgraph UI["🖥️ Streamlit Tabs"]
        J --> T1["📖 Preview"]
        J --> T2["🧩 Plan"]
        J --> T3["🔎 Research"]
        J --> T4["🖼️ Images"]
        J --> T5["🪲 Debug Logs"]
    end

    style MAIN fill:#1a1e25,stroke:#2e3440,color:#e8eaf0
    style RESEARCH fill:#13161b,stroke:#7a9ee8,color:#e8eaf0
    style WORKERS fill:#13161b,stroke:#5cb88a,color:#e8eaf0
    style REDUCER fill:#13161b,stroke:#e8c97a,color:#e8eaf0
    style UI fill:#13161b,stroke:#2e3440,color:#e8eaf0
```

### Agent Roles

| Agent | Responsibility |
|---|---|
| **Router** | Decides whether web research is needed (`closed_book` / `hybrid` / `open_book`) and generates search queries |
| **Research** | Runs Tavily web searches, deduplicates and validates evidence items |
| **Orchestrator** | Produces a structured `Plan` (4–6 sections) with goals, bullets, word targets, and flags |
| **Worker (×N)** | Writes one section each in parallel, grounded by evidence and plan constraints |
| **Critic** | Scores each section's groundedness against Evidence, flags unsupported claims, and triggers one bounded self-correction rewrite |
| **Reducer** | Merges sections → runs the Critic → decides images → generates images via Flux → embeds into final Markdown |

---

##  Features

- **Intelligent routing** — automatically selects `closed_book`, `hybrid`, or `open_book` mode based on topic volatility
- **Web research** — Tavily-powered search with LLM-driven deduplication and source quality filtering
- **Long-term memory** — every generated blog (topic, tone, quality score) is persisted to `memory_store.json`; the Orchestrator recalls topically-related past runs so it avoids repeating titles/structure and stays tone-consistent across sessions
- **Human-in-the-loop approval gate** — the pipeline can pause after planning so a human reviews/edits section titles, bullets, word targets, and citation/code flags (via an editable table in the Streamlit UI) before any writer agent runs; an "auto mode" toggle skips the pause for single-click generation
- **Reflection / self-critique loop** — a Critic agent scores each citation-requiring or `open_book` section for groundedness (0–100) against the Evidence pool and triggers one bounded rewrite pass for any section that scores below 70, flagging unsupported claims
- **Parallel writing** — LangGraph `Send()` fans out section tasks to worker agents concurrently
- **AI image generation** — Free Flux model via Pollinations API; images get professional text overlays with title/subtitle
- **Structured outputs** — All agents use Pydantic schemas (`Plan`, `Task`, `EvidencePack`, `ImageSpec`, `ClaimCheck`) for reliable, validated responses
- **Graceful degradation** — Image, research, and critic-judge failures are caught and logged without blocking the pipeline
- **Streamlit UI** — Dark editorial interface with live preview, plan editor, evidence viewer, evaluation scores, memory log, image gallery, and debug logs
- **Download options** — Export as `.md` or a bundled `.zip` (Markdown + images folder)
- **Offline test suite** — `test_pipeline_offline.py` exercises the full graph (routing, memory recall, planning, human-edit simulation, parallel fanout, self-correction, image placement, memory persistence) with every external LLM/search/image call mocked, so the pipeline logic can be verified with zero API cost or network access

---

##  UI Tabs

| Tab | Contents |
|---|---|
| **Preview** | Rendered blog with inline images and download buttons |
| **Content Plan** | Editable content plan table (sections, goals, word targets, citation/code flags) — doubles as the human-in-the-loop approval screen |
| **Research** | Evidence sources table (title, source domain, URL) |
| **Evaluation** | Per-section groundedness scores from the Critic agent, overall quality score, and flagged unsupported claims |
| **Memory** | Long-term memory log of every blog generated across sessions, plus which past blogs were recalled for the current topic |
| **Images** | Gallery of AI-generated images |
| **Debug** | Full JSON debug dump of pipeline state |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) (StateGraph + subgraph) |
| LLM | [Groq](https://groq.com/) — `llama-3.3-70b-versatile` |
| Web Search | [Tavily](https://tavily.com/) |
| Image Generation | [Pollinations.ai](https://pollinations.ai/) (Flux model, free) |
| Image Processing | Pillow (text overlay rendering) |
| Data Validation | Pydantic v2 |
| UI | Streamlit |
| Environment | python-dotenv |

---

## 📁 Project Structure

```
blogforge-ai/
├── bwa_research_image.py      # Core multi-agent pipeline (LangGraph)
├── streamlit_blog.py          # Streamlit frontend UI
├── memory_store.py            # Lightweight long-term memory (JSON-based recall)
├── test_pipeline_offline.py   # Offline end-to-end test (all external calls mocked)
├── memory_store.json          # Generated at runtime — long-term memory (not committed)
├── images/                    # Generated images (auto-created at runtime)
├── .env                       # API keys (not committed)
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Pakhi27/Multi-Agent-AI-Content-Generation-System.git
cd Multi-Agent-AI-Content-Generation-System
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

| Variable | Where to get it |
|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) |
| `TAVILY_API_KEY` | [app.tavily.com](https://app.tavily.com) |

### 4. Run the Streamlit app

```bash
streamlit run streamlit_blog.py
```

### 5. (Optional) Run the pipeline from the CLI

```bash
python bwa_research_image.py
```

This will generate a blog on `"Self Attention in Transformer Architecture"` by default and print the final Markdown to stdout.

---

##  How It Works — Step by Step

1. **Router** receives the topic and determines research mode + generates search queries if needed.
2. **Research Agent** (optional) queries Tavily, cleans results, and builds a deduplicated `EvidencePack`.
3. **Memory recall** — the Orchestrator looks up topically-related past blogs from long-term memory before planning.
4. **Orchestrator** creates a structured `Plan` — 4 to 6 sections with goals, bullets, word targets, and flags like `requires_code` or `requires_citations`.
5. **Human-in-the-loop checkpoint** *(optional, on by default)* — the pipeline pauses here; you review/edit the plan in the Streamlit UI, then approve it to continue.
6. **Fan-out** dispatches each section as a parallel `Send()` to individual Worker agents.
7. **Workers** write their sections in Markdown, grounded by the plan and evidence, respecting citation and code constraints.
8. **Critic (self-correction loop)** grades each citation-requiring/`open_book` section's groundedness against the evidence and rewrites any section scoring below 70, once.
9. **Reducer subgraph** merges all sections in order, decides on up to 2 image placements, generates images via Flux, adds text overlays, and embeds them into the final Markdown document.
10. The final `.md` file is saved to disk, the run is written to long-term memory, and everything is rendered in the Streamlit UI.

---

##  Testing

Run the offline end-to-end test suite — it mocks every Groq/Tavily/image-API call so the full graph (routing, memory recall, planning, human-edit simulation, parallel fan-out, self-correction, image placement, memory persistence) can be verified with zero API cost and no network access:

```bash
python test_pipeline_offline.py
```

---

##  Requirements

```
langgraph
langchain-groq
langchain-core
langchain-community
tavily-python
pydantic
pillow
requests
streamlit
pandas
python-dotenv
```

> Generate a `requirements.txt` with `pip freeze > requirements.txt` after installing.

---

##  Configuration & Customisation

**Change the default topic** in `bwa_research_image.py`:
```python
if __name__ == "__main__":
    result = run("Your topic here")
```

**Adjust section count** — edit the `Plan` schema:
```python
tasks: List[Task] = Field(..., min_length=4, max_length=6)
```

**Swap the LLM model** — update the Groq model name:
```python
llm_strong = ChatGroq(model="llama-3.3-70b-versatile", ...)
```

**Image dimensions** — change the Pollinations URL in `generate_flux_image()`:
```python
url = f"...?width=1280&height=720&model=flux..."
```

---

##  Known Limitations

- Image generation depends on the free Pollinations API — availability and quality may vary
- LLaMA 3.3 on Groq has rate limits; large plans with many parallel workers may occasionally hit them (retries are configured)
- Font rendering for image overlays requires `arial.ttf` to be available on the system; falls back to PIL's default font otherwise

---

##  License

MIT License — feel free to use, modify, and distribute.

---

##  Acknowledgements

- [LangChain / LangGraph](https://github.com/langchain-ai/langgraph) for the agent orchestration framework
- [Groq](https://groq.com/) for fast LLaMA inference
- [Tavily](https://tavily.com/) for research-grade web search
- [Pollinations.ai](https://pollinations.ai/) for free Flux image generation
