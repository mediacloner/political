# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A local-only, multi-agent geopolitical debate system where three specialized LLM instances represent distinct geopolitical perspectives (US, China, EU) and engage in structured debates. Output can be rendered as a podcast via TTS.

**Hardware target:** NVIDIA RTX 3060 12 GB VRAM · 32 GB system RAM · No cloud dependencies.

## Current State

The project is in the **architecture/planning phase**. The file `AI Swarm - Improvements Report.md` is the canonical design document — read it before implementing anything. There is no source code yet.

## Planned Tech Stack

- **Inference:** ExLlamaV2 + TabbyAPI (replaces Ollama)
- **Quantization:** EXL2 variable bitrate ~4.0 bpw (replaces GGUF Q4_K_M)
- **Orchestration:** Custom Python (~400 lines), not CrewAI or LangGraph
- **Search:** DuckDuckGo (`duckduckgo-search`) + Trafilatura; Tavily as fallback
- **Embeddings/RAG:** `sentence-transformers` (all-MiniLM-L6-v2, CPU) + ChromaDB or FAISS
- **TTS:** Fish Speech (primary) + Dia (2-speaker dialogue exchanges)
- **Language:** Python

## Model Roster

| Agent | Model | Context | Role |
|---|---|---|---|
| US Delegation | Gemma 3 12B (EXL2) | 128K | Debater — US perspective |
| China Delegation | DeepSeek-R1-Distill-Qwen-14B (EXL2) | 128K | Debater — Chinese perspective |
| EU Judge | Qwen 2.5 14B (EXL2) | 128K | Synthesizer / Evaluator |

Only one model fits in VRAM at a time. The orchestrator must hot-swap via TabbyAPI (`POST /v1/model/unload` → `POST /v1/model/load`). Pre-load all three into system RAM for 1–3s swap time instead of 5–15s from SSD.

## Architecture Decisions

**Orchestration:** Use a custom Python debate loop, not a framework. The loop: load model → inject context → generate → extract `<argument>` from CoT response → update debate state → swap model → repeat.

**Context per turn (~1,500–3,000 tokens total):**
- Tier 1: Persona/character sheet (~300–500 tokens, always present)
- Tier 2: Debate state + key claims (~200–400 tokens)
- Tier 3: Last 2–3 verbatim turns (~500–1,500 tokens)
- Tier 4: Hierarchical summaries of older turns (~300–800 tokens)

For debates >10 rounds, add RAG over prior turns using ChromaDB/FAISS.

**Prompt engineering:**
- Debaters: temperature 0.8–1.0; judge: 0.3–0.5
- Require explicit disagreement and adversarial framing to prevent position collapse
- Use `<thinking>` / `<argument>` sections so CoT reasoning is hidden from the debate log
- Re-establish full persona each invocation (do not rely on conversation history for identity)

**Quality evaluation (post-round):**
- LLM-as-judge scores each turn on novelty, evidence, engagement, coherence
- Embedding similarity (all-MiniLM-L6-v2) detects repetition without GPU
- Claim tracker: JSON document of assertions, agreements, contentions, evidence

## Implementation Priority

| Priority | Task |
|---|---|
| P0 | Download Gemma 3 12B + Qwen 2.5 14B, quantize to EXL2 |
| P0 | Set up ExLlamaV2 + TabbyAPI |
| P1 | Custom Python orchestrator with model hot-swap |
| P1 | Tiered context management + summarization |
| P1 | Adversarial persona prompt templates |
| P2 | RAM pre-loading (double-buffer all 3 models) |
| P2 | Web research (DuckDuckGo + Trafilatura) |
| P2 | Fish Speech TTS pipeline |
| P3 | Debate quality scoring + repetition detection |
| P3 | RAG for long debates (ChromaDB/FAISS) |
