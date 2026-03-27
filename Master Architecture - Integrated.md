# Master Architecture Document: Geo-Distributed AI Swarm
## Integrated Design & Implementation Reference

> **Version:** 2.0 — Integrated
> **Date:** 2026-03-27
> **Status:** Architecture/Planning Phase — No source code yet
> **Source documents:** `AI sawrn.MD` (original design) + `AI Swarm - Improvements Report.md` (component upgrades)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Hardware Specifications & Constraints](#2-hardware-specifications--constraints)
3. [Multi-Agent Roster](#3-multi-agent-roster)
4. [Software Stack](#4-software-stack)
5. [Operational Flow — 4 Phases](#5-operational-flow--4-phases)
6. [Context Management](#6-context-management)
7. [Prompt Engineering](#7-prompt-engineering)
8. [Web Research Pipeline](#8-web-research-pipeline)
9. [TTS / Podcast Production](#9-tts--podcast-production)
10. [Quality & Evaluation](#10-quality--evaluation)
11. [Performance Optimizations](#11-performance-optimizations)
12. [Case Study: The Space Race Summit](#12-case-study-the-space-race-summit)
13. [Implementation Priority](#13-implementation-priority)
14. [Before vs After — Component Summary](#14-before-vs-after--component-summary)

---

## 1. Executive Summary

A fully local, multi-agent geopolitical debate system where three specialized LLM instances represent distinct geopolitical perspectives (US, China, EU) and engage in structured, adversarial debates. The system supports dynamic web research, iterative cross-examination, and automated quality evaluation. Output is rendered as a structured transcript and optionally as a multi-speaker audio podcast via TTS.

**Design goals:**
- Zero cloud dependencies — absolute data privacy, zero API costs
- Fit entirely within 12 GB VRAM via sequential hot-swapping
- Produce high-quality, non-collapsing adversarial debate with sustained disagreement
- Scale to 20+ debate rounds without context overflow

**Hardware target:** NVIDIA RTX 3060 · 12 GB VRAM · 32 GB system RAM

---

## 2. Hardware Specifications & Constraints

| Resource | Spec | Role |
|---|---|---|
| GPU | NVIDIA RTX 3060 | Inference (one model at a time) |
| VRAM | 12 GB | Hard ceiling — dictates sequential execution |
| System RAM | 32 GB | Pre-load all 3 models (~25 GB total) for fast swapping |
| Storage | SSD | Initial model load; RAM pre-loading eliminates repeat disk reads |

**Execution strategy — Sequential Hot-Swapping:**
Only one model lives in VRAM at any time. When an agent's turn ends, the orchestrator calls `POST /v1/model/unload`, then `POST /v1/model/load` for the next agent. With all three models pre-loaded into RAM, swap time drops from 5–15 s (disk→GPU) to **1–3 s (RAM→GPU)**.

---

## 3. Multi-Agent Roster

### Node A: United States Delegation

| Attribute | Value |
|---|---|
| **Model** | Gemma 3 12B (Google, March 2025) |
| **Quantization** | EXL2 ~4.5 bpw |
| **VRAM footprint** | ~7.5 GB |
| **Context window** | 128K tokens |
| **MMLU** | ~75% |
| **Temperature** | 0.8–1.0 |

**Dynamic personas:**
- US Geopolitical Strategist
- US Tech Corporate Lobbyist
- US Financial Analyst

**Why Gemma 3 12B over Gemma 2 9B:** 16× larger context window (128K vs 8K), +4 MMLU points, comparable VRAM. The Gemma 2's 8K context was the tightest bottleneck in the original design — it couldn't hold more than 3–4 debate turns.

---

### Node B: Chinese Delegation

| Attribute | Value |
|---|---|
| **Model** | DeepSeek-R1-Distill-Qwen-14B (DeepSeek / Alibaba) |
| **Quantization** | EXL2 ~3.75 bpw |
| **VRAM footprint** | ~8.0 GB |
| **Context window** | 128K tokens |
| **MMLU** | ~79% |
| **Temperature** | 0.8–1.0 |

**Dynamic personas:**
- PRC Strategic Affairs Director
- State-Owned Enterprise Representative
- Macro-Economist

**Why keep this model:** The R1 distillation gives it chain-of-thought reasoning inherited from the 671B DeepSeek-R1. No other 14B-class model matches its analytical depth for first-principles geopolitical reasoning. The plain Qwen 2.5 14B alternative trades reasoning depth for instruction following — a net loss for a debate agent.

---

### Node C: European Synthesiser (The Judge)

| Attribute | Value |
|---|---|
| **Model** | Qwen 2.5 14B Instruct (Alibaba) |
| **Quantization** | EXL2 ~3.5 bpw |
| **VRAM footprint** | ~8.5 GB |
| **Context window** | 128K tokens |
| **MMLU** | ~80% |
| **MT-Bench** | ~8.5+ |
| **Temperature** | 0.3–0.5 |

**Dynamic persona:** European Commission Lead Strategist

**Why Qwen 2.5 14B over Mistral Nemo 12B:** +7 MMLU points (80% vs 73%), significantly stronger synthesis and structured output. The judge must be the smartest agent — it identifies logical flaws, weighs conflicting evidence, and produces balanced verdicts. At 3.5 bpw EXL2, it leaves ~3 GB for KV cache and CUDA overhead.

> **Rejected alternative — Phi-4 14B:** ~84% MMLU but only 16K context window — a dealbreaker for a judge that must ingest the entire debate transcript.

---

## 4. Software Stack

### Inference Engine: ExLlamaV2 + TabbyAPI

Replaces Ollama. Key improvement: ~2× faster token generation, API-driven model management (no process restarts), and native EXL2 quantization support.

```
POST /v1/model/unload          # evict current model from VRAM
POST /v1/model/load            # load next agent's model
  { "model": "agent-model", "max_seq_len": 4096 }
POST /v1/chat/completions      # OpenAI-compatible inference call
```

| Metric | Ollama | ExLlamaV2 + TabbyAPI |
|---|---|---|
| Token speed (13B, 4-bit) | 10–20 t/s | **25–40 t/s** |
| Model swap | Process restart via `keep_alive=0` | **API endpoints** |
| Quantization formats | GGUF only | **EXL2 (native), GPTQ, GGUF** |
| VRAM control | All-or-nothing | **Per-layer GPU/CPU split** |

### Orchestration: Custom Python (~400 lines)

Replaces CrewAI / LangGraph. The debate loop is a fixed 3-agent cycle — no framework abstraction provides value here that outweighs the dependency cost.

```python
for round in range(max_rounds):
    for agent in [us_agent, china_agent, europe_judge]:
        swap_model(agent.model_name)            # TabbyAPI load/unload
        response = generate(agent.prompt(state)) # OpenAI-compatible call
        state.add_turn(agent.name, response)
    if judge_says_done(state):
        break
```

Zero external dependencies beyond `requests`. Full hot-swap control. No framework API churn risk.

### Web Research: DuckDuckGo + Trafilatura + Tavily fallback

| Component | Tool | Purpose |
|---|---|---|
| Search | `duckduckgo-search` | Find relevant URLs, no API key |
| Extraction | Trafilatura | Extract clean article text locally |
| Fallback search | Tavily (free tier, 1,000/mo) | When DDG rate-limits |
| JS-heavy pages | Jina Reader (`r.jina.ai/{url}`) | Markdown from JS-rendered pages |

### Embeddings / RAG: sentence-transformers + ChromaDB or FAISS

- Model: `all-MiniLM-L6-v2` (~80 MB, CPU-only, zero VRAM cost)
- Used for: repetition detection (cosine similarity between turns) and RAG retrieval for debates >10 rounds

### TTS: Fish Speech (primary) + Dia (dialogue segments)

Replaces XTTSv2 (unmaintained since Coqui shutdown, late 2023) and Kokoro-82M (no voice cloning).

| Engine | Quality | Voice Cloning | Multi-Speaker | VRAM | Speed | License |
|---|---|---|---|---|---|---|
| **Fish Speech** | Very Good | Yes (zero-shot) | Yes | ~2–4 GB | 1–3× RT | Apache 2.0 |
| Dia (Nari Labs) | Very Good | Limited | Native 2-speaker | ~4–6 GB | 0.5–1× RT | Apache 2.0 |
| XTTSv2 (old) | Very Good | Yes | Yes | ~2–4 GB | 0.5–1× RT | Restrictive |

**Strategy:** Use Fish Speech for full 3-speaker generation. Optionally use Dia for US↔China dialogue segments (native 2-speaker turn-taking), then Fish Speech for EU verdict narration.

---

## 5. Operational Flow — 4 Phases

### Phase 1: Research & Initial Proposals

1. Load US model → inject *US Lobbyist* persona → web research (DDG + Trafilatura) → generate position paper → extract `<argument>` → unload.
2. Load China model → inject *PRC Director* persona → web research → generate counter-proposal → extract `<argument>` → unload.
3. Update `debate_state` JSON with initial claims and evidence.

### Phase 2: The Iterative Debate Loop (Time-Bounded)

System enters a loop bounded by a user-defined time limit (e.g., 20 minutes, or `max_rounds`).

```
while elapsed < time_limit and round < max_rounds:
    1. Judge reviews transcript → generates targeted question for weakest argument
    2. Challenged agent reloads → researches rebuttal → responds
    3. Opponent agent reloads → counter-argues
    4. Update debate_state, run repetition check, run quality scoring
    5. If cosine similarity > 0.85 for 3 consecutive turns → break early
    6. If judge signals consensus or exhaustion → break early
```

At 18-minute mark (configurable), the loop exits regardless of round count to ensure the verdict phase has time to complete.

### Phase 3: Verdict & Transcript Generation

1. Load EU Judge → inject full tiered context (persona + debate state + compressed history).
2. Judge runs structured rubric: steelman both sides → score each dimension → identify blind spots → deliver verdict.
3. Save complete transcript to disk as JSON + Markdown.
4. Save `debate_state` JSON (claims, evidence, agreements, contentions).

### Phase 4: Podcast Production

1. All LLM models purged from VRAM.
2. Summarisation script converts raw transcript → natural dialogue script with 3 speaker labels.
3. Fish Speech loaded → assign distinct voice clones (10–30s reference clips per speaker).
4. Generate audio sequentially per speaker turn → compile into final `.wav`.

---

## 6. Context Management

### Tiered Context Window (per agent turn, ~1,500–3,000 tokens total)

| Tier | Content | Token Budget | Persistence |
|---|---|---|---|
| **1 — Persona** | System prompt + character sheet + debate rules | ~300–500 | Always |
| **2 — Debate State** | Running JSON: claims, evidence, contentions | ~200–400 | Updated each round |
| **3 — Recent History** | Last 2–3 verbatim turns | ~500–1,500 | Sliding window |
| **4 — Compressed Past** | Hierarchical summary of older turns | ~300–800 | Grows logarithmically |

### Hierarchical Summarization

```
Turns 1-3  →  Round 1 Summary  (~100-150 tokens)
Turns 4-6  →  Round 2 Summary  (~100-150 tokens)
Turns 7-9  →  Round 3 Summary  (~100-150 tokens)

Round 1-3 Summaries  →  Phase 1 Summary  (~100-150 tokens)
```

Memory usage scales **O(log n)** relative to debate length. A 20-round debate uses roughly the same context budget as a 5-round debate.

### RAG for Long Debates (10+ rounds)

Store all turns in ChromaDB or FAISS. At generation time, retrieve top-K most semantically relevant prior turns via `all-MiniLM-L6-v2` embeddings. Zero VRAM cost (CPU-only).

---

## 7. Prompt Engineering

### 7A. Persona Anchoring Template

Each invocation is stateless — the model has no memory beyond what is injected. Every prompt must fully re-establish the persona.

```
You are {name}, {title}.

CORE BELIEFS:
- {belief_1}
- {belief_2}
- {belief_3}

DEBATE STYLE: {aggressive/measured/Socratic}
RHETORICAL APPROACH: {data-driven/appeals to precedent/first-principles}

FORBIDDEN:
- Never concede {core_position} without overwhelming new evidence.
- Never use ad hominem attacks.
- Never break character.

CURRENT DEBATE: {topic}
YOUR OBJECTIVE THIS TURN: {specific instruction for this turn}
```

### 7B. Preventing Collapse Into Agreement

The single biggest quality risk in multi-agent debate. LLMs are trained to be agreeable — this fights sustained disagreement.

**Countermeasures (use all):**
1. **Adversarial instruction:** "Your PRIMARY goal is to identify the weakest point in your opponent's last argument and attack it."
2. **Disagreement requirement:** "Your response MUST contain at least one point of explicit disagreement with the previous speaker."
3. **Structural conflict:** Assign ideologically opposed frameworks (realism vs. liberal internationalism, state-capitalism vs. free-market) that make agreement structurally impossible.
4. **Temperature:** 0.8–1.0 for debaters, 0.3–0.5 for judge.
5. **Periodic devil's advocate injection:** Every 3–4 turns: "The debate is converging. Introduce a controversial counterpoint that challenges the emerging consensus."

### 7C. Hidden Chain-of-Thought

```
<thinking>
[Internal strategic reasoning — hidden from other agents and debate log]
</thinking>

<argument>
[The actual debate turn — passed to other agents and the judge]
</argument>
```

Only `<argument>` enters the shared transcript. `<thinking>` improves argument quality via CoT without cluttering the debate history.

### 7D. Judge Rubric

```
Before giving your final assessment:
1. Steelman both sides — present the strongest version of each argument.
2. Score each side on: Evidence Quality (1-5), Logical Coherence (1-5),
   Persuasiveness (1-5), Engagement with Counterarguments (1-5).
3. Identify perspectives that NEITHER debater addressed.
4. Only then deliver your synthesis and verdict.

Do NOT manufacture false balance. If one side is clearly stronger, say so.
```

---

## 8. Web Research Pipeline

```
Query → DuckDuckGo (duckduckgo-search)
      → Returns URLs + snippets
      → Trafilatura extracts full article text locally
      → If JS-heavy page → Jina Reader (r.jina.ai/{url})
      → If DDG rate-limited → Tavily fallback
      → Clean text passed to agent context
```

Agents receive full article text, not just search snippets — enabling deep analysis of source material rather than surface-level referencing.

---

## 9. TTS / Podcast Production

**Pipeline:**
1. Transcript → summarisation script → 3-speaker dialogue script
2. Assign voice reference clips (10–30s per speaker)
3. Fish Speech generates audio per turn
4. Optional: use Dia for US↔China exchange segments (native 2-speaker dialogue with natural turn-taking)
5. Concatenate audio → final `.wav`

**Voice differentiation strategy:**
- US Delegation: authoritative, measured pace
- China Delegation: precise, deliberate cadence
- EU Judge: neutral, analytical tone

---

## 10. Quality & Evaluation

### 10A. LLM-as-Judge (after each round)

```
Evaluate the last round. Score each participant:
- Argument novelty (1-5): New points or repetition of prior claims?
- Evidence quality (1-5): Specific data, events, or sources cited?
- Engagement (1-5): Did they address the opponent's arguments directly?
- Logical coherence (1-5): Fallacies or internal contradictions?
```

Use score trajectory to detect stagnation and trigger early termination instead of relying solely on time cutoffs.

### 10B. Repetition Detection (CPU, zero VRAM)

After each turn, compute semantic embedding with `all-MiniLM-L6-v2` and compare against all prior turns from the same agent.

- Cosine similarity > 0.85 with any prior turn → flag as repetitive
- 3 consecutive flagged turns → end debate early (arguments exhausted)

### 10C. Claim Tracking

```json
{
  "us_claims": ["claim1", "claim2"],
  "china_claims": ["claim1", "claim2"],
  "points_of_agreement": ["..."],
  "points_of_contention": ["..."],
  "evidence_cited": {
    "us": ["source1", "source2"],
    "china": ["source1", "source2"]
  }
}
```

Updated after each turn. Feeds both the context management system (Tier 2) and the judge's final synthesis.

---

## 11. Performance Optimizations

### RAM Pre-loading (Double-Buffering)

Pre-load all three models into system RAM at startup. VRAM swaps read from RAM, not SSD.

```
Current:   [===DISK→GPU 8-15s===][==Generate 30s==][=Unload=][===DISK→GPU 8-15s===]
Proposed:  [=RAM→GPU 1-3s=][==Generate 30s==][=RAM→GPU 1-3s=][==Generate 30s==]
```

Time saved per full debate round (3 swaps): ~15–35 seconds.

### KV Cache Quantization

Quantize KV cache from FP16 to Q8 (virtually lossless) or Q4 (minor perplexity increase, acceptable for debate).
- Saves 1–2 GB VRAM for 14B model at 4K context
- Critical headroom when largest model takes ~8.5 GB

### Speculative Decoding (Optional / P3)

Pair each model with a ~0.5–1B draft model. Expected 1.3–1.8× speedup for debate-style text. VRAM cost ~0.5–1 GB. Worth experimenting with after core system is stable.

---

## 12. Case Study: The Space Race Summit

**Debate prompt:** *"Should the European Space Agency (ESA) partner with NASA's Artemis programme or the CNSA's International Lunar Research Station (ILRS) for the establishment of a permanent lunar base?"*

### Phase 1 — Research & Proposals

1. US Model loaded as *US Tech Corporate Lobbyist* → researches current Artemis contracts, commercial aerospace partners → generates proposal emphasizing commercial benefits and democratic technology standards → `<argument>` extracted → model unloaded.
2. China Model loaded as *PRC Strategic Affairs Director* → researches Chang'e programme timeline, ILRS partners → generates proposal emphasizing resource efficiency, guaranteed state-backed funding, and non-interference principles → `<argument>` extracted → model unloaded.
3. `debate_state` initialized with initial claims from both sides.

### Phase 2 — Iterative Debate Loop

*Timer: 20 minutes. Loop begins.*

**Round 1:**
- EU Judge reviews proposals → identifies financial discrepancy → generates targeted question: *"US Analyst, how do you mitigate the risk of congressional budget cuts delaying the Artemis timeline compared to China's state-backed funding certainty?"*
- US Model reloaded → researches recent congressional space budgets → counter-argues with multi-decade NASA budget stability data → unloaded.
- China Model reloaded → challenges counter-argument with SpaceX dependency risk → unloaded.
- Quality scoring: novelty high, engagement high. No repetition detected. Continue.

*Rounds 2–6: Cross-examination continues on technology transfer risk, lunar resource rights, ESA autonomy.*

*At 18-minute mark: loop exits.*

### Phase 3 — Verdict

EU Judge (Qwen 2.5 14B) ingests full tiered context. Applies structured rubric:
1. Steelmans both sides
2. Scores evidence quality, coherence, persuasiveness, engagement
3. Identifies blind spot: neither side addressed ESA's existing dual-use technology agreements
4. Delivers verdict: conditional partnership with NASA, with specific safeguards and an ILRS observer status clause

Transcript saved as JSON + Markdown.

### Phase 4 — Podcast

Transcript → 3-speaker dialogue script → Fish Speech generates audio → final `.wav` compiled.
Runtime ~5 minutes audio from a 20-minute debate.

---

## 13. Implementation Priority

| Priority | Task | Effort | Impact |
|---|---|---|---|
| **P0** | Download Gemma 3 12B + Qwen 2.5 14B, quantize to EXL2 | Low (download + quant) | High |
| **P0** | Install ExLlamaV2 + TabbyAPI, verify model load/unload API | Medium (install + config) | High |
| **P1** | Custom Python orchestrator with hot-swap loop | Medium (~400 lines) | High |
| **P1** | Tiered context management + hierarchical summarization | Medium | High |
| **P1** | Adversarial persona prompt templates + hidden CoT format | Low (prompt design) | High |
| **P2** | RAM pre-loading (pre-load all 3 models at startup) | Low (config) | Medium |
| **P2** | Web research pipeline (DDG + Trafilatura + Tavily fallback) | Low (`pip install`) | Medium |
| **P2** | Fish Speech TTS pipeline + voice reference setup | Medium | Medium |
| **P3** | LLM-as-judge quality scoring after each round | Medium | Medium |
| **P3** | Embedding-based repetition detection (all-MiniLM-L6-v2) | Low | Medium |
| **P3** | RAG over prior turns for debates >10 rounds (ChromaDB/FAISS) | Medium | Low-Medium |
| **P3** | Speculative decoding experiment | High | Low-Medium |

---

## 14. Before vs After — Component Summary

| Component | Original Design | Updated Design | Key Gain |
|---|---|---|---|
| **Inference engine** | Ollama + `keep_alive=0` | ExLlamaV2 + TabbyAPI | 2× faster, API-driven swap |
| **US model** | Gemma 2 9B (8K ctx, MMLU ~71%) | Gemma 3 12B (128K ctx, MMLU ~75%) | 16× context, better reasoning |
| **China model** | DeepSeek-R1-Distill-Qwen-14B | Same (keep) | Already optimal |
| **EU judge model** | Mistral Nemo 12B (MMLU ~73%) | Qwen 2.5 14B (MMLU ~80%) | +7 MMLU, stronger synthesis |
| **Quantization** | GGUF Q4_K_M (~4.8 bpw) | EXL2 variable (~3.5–4.5 bpw) | Better quality per bit, custom sizing |
| **Orchestration** | CrewAI / LangGraph | Custom Python (~400 lines) | Zero deps, full hot-swap control |
| **Model swap time** | 5–15 s (disk→GPU) | 1–3 s (RAM→GPU) | 5–10× faster |
| **Context strategy** | None (full transcript) | Tiered window + summarization | O(log n) memory, no overflow |
| **Prompt engineering** | Basic personas | Adversarial system + hidden CoT | Prevents agreement collapse |
| **Web research** | DuckDuckGo snippets only | DDG + Trafilatura + Tavily | Full article text extraction |
| **TTS engine** | XTTSv2 (unmaintained) | Fish Speech (Apache 2.0, active) | Maintained, faster, permissive |
| **Quality control** | None | LLM-as-judge + cosine similarity | Detects stagnation, ends early |

---

*This document integrates `AI sawrn.MD` (original architecture) and `AI Swarm - Improvements Report.md` (component upgrades). It supersedes both as the single authoritative reference for implementation.*
