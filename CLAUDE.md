# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A fully local, multi-agent geopolitical debate system where three specialized LLM instances represent distinct geopolitical perspectives (US, China, EU) and engage in structured debates. Includes a web dashboard for real-time monitoring and podcast generation via Edge TTS.

**Hardware target:** NVIDIA RTX 3060 12 GB VRAM · 32 GB system RAM · No cloud dependencies.

## Current State

The project is **fully implemented and functional**. All core systems work end-to-end: research, debate, verdict, podcast, web dashboard.

## Tech Stack

- **Inference:** ExLlamaV2 + TabbyAPI (submodule at `tabbyAPI/`)
- **Quantization:** EXL2 variable bitrate ~3.5–4.0 bpw
- **Orchestration:** Custom Python debate loop (`src/orchestrator.py`)
- **Search:** DuckDuckGo (`ddgs`) + Trafilatura + Jina Reader; Tavily as fallback
- **Embeddings/RAG:** `sentence-transformers` (all-MiniLM-L6-v2, CPU) + ChromaDB
- **TTS:** Edge TTS (Microsoft free neural voices, no GPU needed)
- **Web dashboard:** Flask + vanilla JS (`dashboard.py` + `static/index.html`)
- **Language:** Python

## Model Roster

| Agent | Model | bpw | Role |
|---|---|---|---|
| US Delegation | Gemma 3 12B IT (EXL2) | 4.0 | Debater — US perspective |
| China Delegation | DeepSeek-R1-Distill-Qwen-14B (EXL2) | 3.5 | Debater — Chinese perspective |
| EU Judge | Mistral Nemo Instruct 12B (EXL2) | 4.0 | Synthesizer / Evaluator (European model) |

Models are stored in `tabbyAPI/models/`. Only one model fits in VRAM at a time — the orchestrator hot-swaps via TabbyAPI SSE streaming API.

## Key Architecture

**Debate flow:**
1. Phase 1 — Research: DDG search + article extraction per agent
2. Phase 2 — Debate loop: US → China → EU → score → repeat
3. Phase 3 — Verdict: EU judge delivers structured ruling
4. Phase 4 — Podcast: LLM generates dialogue script → Edge TTS generates MP3

**Entry points:**
- `menu.py` — Interactive CLI menu (auto-activates `.venv`)
- `menu.py --script scripts/foo.yaml` — Run a debate script directly
- `dashboard.py` — Web dashboard on port 8000

**Live status:** The orchestrator writes to `output/.live_debate.json` after every turn. The dashboard polls `/api/live/debate` to render CLI-launched debates in real time.

**Anti-collapse:** Adversarial persona prompts, devil's advocate injection, embedding-based repetition detection, quality scoring with stagnation exit.

**Tag parsing:** Models output `<thinking>` / `<argument>` tags. `BaseAgent.parse_response()` separates them and strips residual tags via regex. DeepSeek-R1 also uses `<think>` tags which are stripped in claim extraction.

## Configuration

- `config/settings.yaml` — TabbyAPI URL, model names/paths, debate params, research settings
- `config/personas.yaml` — Agent identities (3 personas per delegation)
- `tabbyAPI/config.yml` — TabbyAPI server config (`model_dir: models`)
- `scripts/*.yaml` — Pre-configured debate scenarios

## Running

```bash
python menu.py              # CLI menu (auto-activates .venv)
python dashboard.py         # Web dashboard at http://localhost:8000
```

TabbyAPI starts automatically if offline. Models must be downloaded first (stored in `tabbyAPI/models/`).
