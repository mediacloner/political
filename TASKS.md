# Project Tasks — Politics AI Swarm

> **Last updated:** 2026-03-27
> **Reference:** `Master Architecture - Integrated.md` for full specs

---

## P0 — Blockers (Do First)

- [ ] Download Gemma 3 12B from HuggingFace and quantize to EXL2 ~4.5 bpw (~7.5 GB)
- [ ] Download Qwen 2.5 14B Instruct and quantize to EXL2 ~3.5 bpw (~8.5 GB)
- [ ] Verify DeepSeek-R1-Distill-Qwen-14B is available in EXL2 ~3.75 bpw (~8.0 GB)
- [ ] Install ExLlamaV2 (inference backend)
- [ ] Install and configure TabbyAPI (model management server)
- [ ] Test `POST /v1/model/load` and `POST /v1/model/unload` endpoints with each model
- [ ] Verify all 3 models fit in 32 GB RAM simultaneously

---

## P1 — Core Implementation

- [ ] Set up Python project structure (`orchestrator/`, `agents/`, `prompts/`, `context/`, `research/`, `tts/`)
- [ ] Write `orchestrator.py` — main debate loop (~400 lines)
  - [ ] `swap_model(model_name)` via TabbyAPI API calls
  - [ ] `generate(prompt, temperature)` via OpenAI-compatible endpoint
  - [ ] 4-phase flow: Research → Debate → Verdict → Podcast
  - [ ] Time-bounded loop with configurable `max_rounds` and `time_limit`
- [ ] Write `context_manager.py` — tiered context assembly
  - [ ] Tier 1: Persona injection (always present)
  - [ ] Tier 2: `debate_state` JSON serialization
  - [ ] Tier 3: Sliding window of last 2–3 verbatim turns
  - [ ] Tier 4: Hierarchical summarization of older turns
- [ ] Write `agents/` — persona templates for each agent
  - [ ] `us_agent.py` — US Delegation (3 dynamic personas)
  - [ ] `china_agent.py` — China Delegation (3 dynamic personas)
  - [ ] `eu_judge.py` — EU Judge with structured rubric
- [ ] Write `prompts/` — adversarial prompt templates
  - [ ] Persona anchoring template (stateless re-establishment each turn)
  - [ ] Anti-collapse instructions (adversarial framing, disagreement requirements)
  - [ ] Hidden CoT format (`<thinking>` / `<argument>` sections)
  - [ ] Judge rubric template (steelman → score → blind spots → verdict)
- [ ] Write `debate_state.py` — claim tracker
  - [ ] JSON schema: `us_claims`, `china_claims`, `points_of_contention`, `evidence_cited`
  - [ ] Update logic after each agent turn

---

## P2 — Enhancements

- [ ] Write `research/web_search.py` — web research pipeline
  - [ ] DuckDuckGo search via `duckduckgo-search`
  - [ ] Trafilatura for full article text extraction
  - [ ] Jina Reader fallback for JS-heavy pages
  - [ ] Tavily fallback when DDG rate-limits
- [ ] Implement RAM pre-loading at orchestrator startup
  - [ ] Pre-load all 3 EXL2 models into system RAM
  - [ ] Verify 1–3s VRAM swap time
- [ ] Write `tts/podcast.py` — podcast production pipeline
  - [ ] Install Fish Speech
  - [ ] Record/source 10–30s voice reference clips for 3 speakers
  - [ ] Transcript → 3-speaker dialogue script conversion
  - [ ] Audio generation + concatenation → final `.wav`
- [ ] KV cache quantization config (Q8 or Q4 in TabbyAPI config)

---

## P3 — Polish & Evaluation

- [ ] Write `evaluation/quality_scorer.py` — LLM-as-judge per round
  - [ ] Score: novelty, evidence quality, engagement, coherence (1–5 each)
  - [ ] Stagnation detection → early termination trigger
- [ ] Write `evaluation/repetition_detector.py` — embedding similarity
  - [ ] Install `sentence-transformers`, load `all-MiniLM-L6-v2` (CPU)
  - [ ] Cosine similarity > 0.85 → flag turn as repetitive
  - [ ] 3 consecutive flags → end debate early
- [ ] Write `rag/retriever.py` — RAG for long debates (10+ rounds)
  - [ ] Install ChromaDB or FAISS
  - [ ] Store all turns as embeddings
  - [ ] Retrieve top-K relevant turns at generation time
- [ ] Experiment with speculative decoding (draft model pairing)
- [ ] Write `tests/` — integration tests for each subsystem
- [ ] Write CLI entrypoint (`main.py`) with argument parsing
  - [ ] `--topic`, `--rounds`, `--time-limit`, `--output-dir`, `--tts`

---

## Documentation

- [x] `AI sawrn.MD` — Original architecture document
- [x] `AI Swarm - Improvements Report.md` — Component upgrade analysis
- [x] `Master Architecture - Integrated.md` — Single authoritative reference (v2.0)
- [x] `CLAUDE.md` — Claude Code project instructions
- [ ] `README.md` — Setup and usage instructions (after P1 complete)
