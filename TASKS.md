# Project Tasks — Politics AI Swarm

> **Last updated:** 2026-03-27
> **Reference:** `Master Architecture - Integrated.md` for full specs

---

## P0 — Blockers (Do First)

- [ ] Download Gemma 3 12B → EXL2 ~4.5 bpw: `python download_models.py --model gemma3`
- [ ] Download Qwen 2.5 14B Instruct → EXL2 ~3.5 bpw: `python download_models.py --model qwen`
- [ ] Download DeepSeek-R1-Distill-Qwen-14B → EXL2 ~3.75 bpw: `python download_models.py --model deepseek`
- [x] Install ExLlamaV2 — `.venv/bin/pip install exllamav2` ✓
- [x] Install and configure TabbyAPI — cloned as submodule, config.yml tuned ✓
- [ ] Start TabbyAPI and verify: `source activate.sh && cd tabbyAPI && python3 main.py`
- [ ] Test model load/unload: `python main.py --check`
- [ ] Verify all 3 models fit in 32 GB RAM simultaneously

---

## P1 — Core Implementation ✅ COMPLETE

- [x] Python project structure (`src/agents/`, `src/context/`, `src/prompts/`, `src/research/`, `src/evaluation/`, `src/rag/`, `src/tts/`)
- [x] `src/tabby_client.py` — TabbyAPI HTTP client (load/unload/swap/chat)
- [x] `src/orchestrator.py` — main debate loop, 4-phase flow, time-bounded, hot-swap
- [x] `src/context/context_manager.py` — tiered context (4 tiers), hierarchical summarization
- [x] `src/context/debate_state.py` — claim tracker, turn history, JSON/Markdown serialization
- [x] `src/prompts/templates.py` — all prompt templates (persona, CoT, judge rubric, summarization)
- [x] `src/agents/base_agent.py` — BaseAgent with parse_response, extract_claims
- [x] `src/agents/us_agent.py` — US Delegation (3 dynamic personas)
- [x] `src/agents/china_agent.py` — China Delegation (3 dynamic personas)
- [x] `src/agents/eu_judge.py` — EU Judge (question + verdict turns)
- [x] `config/settings.yaml` — full system config
- [x] `config/personas.yaml` — all agent persona definitions

---

## P2 — Enhancements ✅ COMPLETE

- [x] `src/research/web_search.py` — DDG + Trafilatura + Jina Reader + Tavily fallback
- [x] RAM pre-loading — TabbyAPI config set; orchestrator pre-loads via `swap_model` at startup
- [x] `src/tts/podcast.py` — Fish Speech pipeline (transcript → script → audio)
- [x] KV cache Q8 — set in `tabbyAPI/config.yml`

---

## P3 — Polish & Evaluation ✅ COMPLETE

- [x] `src/evaluation/quality_scorer.py` — LLM-as-judge per round (4-dimension scoring)
- [x] `src/evaluation/repetition_detector.py` — all-MiniLM-L6-v2, cosine similarity, early exit
- [x] `src/rag/retriever.py` — ChromaDB RAG for 10+ round debates
- [x] `main.py` — full CLI with `--topic`, `--rounds`, `--time-limit`, `--us-persona`, `--china-persona`, `--podcast`, `--check`
- [x] `download_models.py` — HuggingFace EXL2 model downloader

---

## Remaining (User Action Required)

These require hardware/external steps that cannot be automated:

1. **Download models** (requires HuggingFace + ~25 GB disk):
   ```bash
   source activate.sh
   python download_models.py --all
   ```

2. **Start TabbyAPI** (requires models to be present):
   ```bash
   source activate.sh
   cd tabbyAPI && python3 main.py
   ```

3. **Run first debate**:
   ```bash
   source activate.sh
   python main.py --check   # Verify TabbyAPI is up
   python main.py --topic "Should ESA partner with NASA or CNSA for a lunar base?" --rounds 4
   ```

4. **Fish Speech** (optional, for podcast):
   ```bash
   git clone https://github.com/fishaudio/fish-speech
   .venv/bin/pip install -e fish-speech/
   python main.py --topic "..." --podcast --voice-us path/to/us_voice.wav
   ```

---

## Documentation

- [x] `AI sawrn.MD` — Original architecture
- [x] `AI Swarm - Improvements Report.md` — Component upgrade analysis
- [x] `Master Architecture - Integrated.md` — Single authoritative reference (v2.0)
- [x] `CLAUDE.md` — Claude Code project instructions
- [x] `TASKS.md` — This file
