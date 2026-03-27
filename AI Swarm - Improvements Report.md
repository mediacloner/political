# AI Swarm Architecture — Improvements Report

> **Date:** 2026-03-26
> **Scope:** Comprehensive review of every component in the Geo-Distributed AI Swarm architecture, with researched alternatives and justifications for each proposed change.
> **Hardware constraint:** Nvidia RTX 3060 12 GB VRAM · 32 GB System RAM · Local-only execution

---

## Table of Contents

1. [Inference Engine — Replace Ollama with ExLlamaV2 + TabbyAPI](#1-inference-engine)
2. [Model Roster — Upgrade All Three LLMs](#2-model-roster)
3. [Quantization — Switch from GGUF Q4 to EXL2 Variable Bitrate](#3-quantization)
4. [Orchestration Framework — Use Custom Python (or LangGraph)](#4-orchestration-framework)
5. [Performance — RAM Pre-loading & KV Cache Optimizations](#5-performance-optimizations)
6. [Context Management — Hierarchical Summarization Strategy](#6-context-management)
7. [Prompt Engineering — Adversarial Persona System](#7-prompt-engineering)
8. [Web Research — DuckDuckGo + Trafilatura Stack](#8-web-research)
9. [TTS / Podcast — Replace XTTSv2 with Fish Speech (+ Dia for Dialogue)](#9-tts--podcast-production)
10. [Quality & Evaluation — Automated Debate Scoring](#10-quality--evaluation)
11. [Summary: Before vs After](#11-summary-before-vs-after)

---

## 1. Inference Engine

### Current: Ollama with `keep_alive=0`

### Proposed: ExLlamaV2 + TabbyAPI

| Metric | Ollama | ExLlamaV2 + TabbyAPI |
|---|---|---|
| Token speed (7B, 4-bit) | 25–50 t/s | **50–80 t/s** |
| Token speed (13B, 4-bit) | 10–20 t/s | **25–40 t/s** |
| Model swap mechanism | Process-level `keep_alive=0` | **API endpoints** (`POST /v1/model/load`, `/unload`) |
| Swap time (7B from SSD) | 3–8 s | **4–8 s** (comparable, but no process restart) |
| Quantization formats | GGUF only | **EXL2 (native), GPTQ, GGUF (experimental)** |
| VRAM control | Coarse (all-or-nothing) | **Per-layer GPU/CPU split, explicit allocation** |
| OpenAI-compatible API | Yes | Yes |

### Why This Is Better

1. **~2× faster generation.** ExLlamaV2 consistently benchmarks 50–80 t/s for 7B-class models on RTX 3060, versus Ollama's 25–50 t/s. For a debate system running dozens of turns, this saves significant wall-clock time.

2. **API-driven model management.** TabbyAPI exposes dedicated `load` / `unload` endpoints — no process restarts needed. Your orchestrator simply calls:
   ```
   POST /v1/model/unload
   POST /v1/model/load  { "model": "next-agent-model", "max_seq_len": 4096 }
   POST /v1/chat/completions  { ... }
   ```
   This is cleaner and more reliable than Ollama's `keep_alive=0` which depends on timeout-based garbage collection.

3. **EXL2 quantization.** ExLlamaV2's native format allows variable bits-per-weight *per layer*. Critical layers get more bits, redundant layers get fewer. This means better quality at the same file size, or smaller files at the same quality. You can fine-tune each model's footprint to fit comfortably under 12 GB.

4. **Fine-grained VRAM allocation.** You can specify exactly how many layers go to GPU vs CPU for each model, unlike Ollama's all-or-nothing approach. This is critical when your largest model (DeepSeek-R1 14B at ~9 GB) is close to the 12 GB ceiling.

### What You Lose

- **Setup simplicity.** Ollama is `ollama run model` and done. TabbyAPI requires installing ExLlamaV2, downloading EXL2-format models, and configuring a YAML file. This is a one-time cost.
- **Model ecosystem breadth.** GGUF models are more widely available on HuggingFace than EXL2. However, all major models have EXL2 quants available, and you can create your own with ExLlamaV2's quantizer.

### Runner-Up: Ollama (keep current)

If you prioritize simplicity over speed, Ollama remains a solid choice. The `keep_alive=0` approach works. You just leave ~30–50% of potential generation speed on the table.

---

## 2. Model Roster

### Node A: US Delegation

| | Current | Proposed |
|---|---|---|
| **Model** | Gemma 2 9B | **Gemma 3 12B** |
| **VRAM (4-bit)** | ~6.5 GB | **~8 GB** |
| **Context window** | 8K | **128K** |
| **MMLU** | ~71% | **~75%** |

**Why Gemma 3 12B is better:**
- **16× larger context window** (128K vs 8K). This is a massive improvement — the Gemma 2 9B's 8K window was the tightest bottleneck in your entire architecture. It couldn't hold more than ~3–4 debate turns before overflowing.
- **Higher benchmark scores** across reasoning, instruction following, and multilingual tasks.
- **Same VRAM ballpark** (~8 GB vs ~6.5 GB) — still fits comfortably within 12 GB with room for KV cache.
- Released March 2025 by Google; mature and well-supported.

### Node B: China Delegation

| | Current | Proposed |
|---|---|---|
| **Model** | DeepSeek-R1-Distill-Qwen-14B | **DeepSeek-R1-Distill-Qwen-14B (keep)** |
| **VRAM (4-bit)** | ~9 GB | ~9 GB |
| **Context window** | 128K | 128K |
| **MMLU** | ~79% | ~79% |

**Why keep it:** Your current choice is already excellent. The R1 distillation gives it chain-of-thought reasoning capabilities inherited from the full 671B DeepSeek-R1. No other 14B-class model matches its analytical depth for first-principles geopolitical reasoning. The main alternative (plain Qwen 2.5 14B) trades reasoning depth for better instruction following — a net loss for a debate agent.

### Node C: European Judge / Synthesiser

| | Current | Proposed |
|---|---|---|
| **Model** | Mistral Nemo 12B | **Qwen 2.5 14B Instruct** |
| **VRAM (4-bit)** | ~8 GB | **~9–10 GB** |
| **Context window** | 128K | **128K** |
| **MMLU** | ~73% | **~80%** |

**Why Qwen 2.5 14B is better for the judge role:**
- **+7 points on MMLU** (80% vs 73%). The judge needs to be the smartest agent in the room — it must identify logical flaws, weigh conflicting evidence, and produce balanced synthesis. Qwen 2.5 14B is significantly more capable.
- **MT-Bench ~8.5+** — one of the highest instruction-following scores in the 14B class.
- **Same 128K context** as Nemo, so no regression on long-transcript ingestion.
- Outperforms Mistral Nemo on reasoning, analysis, and structured output tasks across benchmarks.
- **VRAM note:** At ~9–10 GB, this is the tightest fit of your three models. With EXL2 quantization you can create a custom 3.5–4.0 bpw quant that targets exactly ~8.5–9 GB, leaving headroom for KV cache.

**Alternative consideration: Phi-4 14B** — scores an astonishing ~84% MMLU and has the best raw reasoning of any 14B model. However, its 16K context window is a dealbreaker for the judge role, which must ingest the entire debate transcript.

---

## 3. Quantization

### Current: GGUF Q4 (4-bit, via Ollama)

### Proposed: EXL2 Variable Bitrate (~4.0 bpw)

| Method | Avg Bits | Quality (Perplexity) | Speed | VRAM Control |
|---|---|---|---|---|
| GGUF Q4_K_M | ~4.8 bpw | Good | Baseline | All-or-nothing per layer |
| GGUF IQ4_XS | ~4.25 bpw | Better than Q4_K_M | Baseline | All-or-nothing |
| GPTQ 4-bit | 4.0 bpw | Good | Good | Uniform |
| AWQ 4-bit | 4.0 bpw | Slightly better than GPTQ | Better than GPTQ | Uniform |
| **EXL2 4.0 bpw** | **Variable (2–8)** | **Best at same size** | **Fastest** | **Per-layer allocation** |

**Why EXL2:**
- **Variable bitrate is the key advantage.** EXL2 assigns more bits to sensitive layers (attention, critical feedforward) and fewer to redundant layers. A 4.0 bpw EXL2 quant consistently outperforms a 4.0 bpw GPTQ or even a ~4.8 bpw GGUF Q4_K_M on perplexity benchmarks.
- **Custom sizing.** You can create quants at exactly the bpw you need. For example:
  - Gemma 3 12B → EXL2 4.5 bpw → ~7.5 GB (comfortable fit)
  - DeepSeek-R1-14B → EXL2 3.75 bpw → ~8.0 GB (tight but safe)
  - Qwen 2.5 14B → EXL2 3.5 bpw → ~8.5 GB (leaves ~3 GB for KV cache + CUDA overhead)
- **Native to ExLlamaV2**, so zero conversion overhead.

### Fallback: GGUF IQ4_XS

If you stay with Ollama/llama.cpp, at minimum upgrade from Q4_K_M to **IQ4_XS** (importance-matrix quantization). It provides better quality at ~4.25 bpw than Q4_K_M at ~4.8 bpw — smaller *and* smarter.

---

## 4. Orchestration Framework

### Current: CrewAI or LangGraph (undecided)

### Proposed: Custom Python orchestration (~300–500 lines)

**Comparison of top options for this specific use case:**

| Criterion | Custom Python | LangGraph | CrewAI | AutoGen/AG2 |
|---|---|---|---|---|
| Model hot-swap control | **Full** | Good (custom nodes) | Poor (manual) | Poor (manual) |
| Debate pattern fit | **Perfect** (you design it) | Good (cyclic graph) | Poor (task-oriented) | Good (GroupChat) |
| State management | **Exactly what you need** | Excellent (typed state) | Awkward (task chaining) | Good (shared history) |
| Complexity to learn | **None** (it's Python) | High (graph theory) | Medium | Medium |
| Dependencies | **Zero** (stdlib + HTTP) | LangChain ecosystem | CrewAI + LiteLLM | AutoGen / AG2 |
| Maintenance risk | **Zero** (you own it) | Low (LangChain Inc.) | Medium (API churn) | **High** (ecosystem split) |

**Why custom Python is the best fit:**

1. **Your use case is specific, not generic.** Framework abstractions (Agent, Task, Crew, Tool) add value when you have many agents doing varied work. You have exactly 3 agents in a fixed debate loop. The framework overhead buys you nothing.

2. **Hot-swapping is the hard part, and no framework handles it.** Every framework requires you to write custom model lifecycle code anyway. If you're already writing the hardest part yourself, why add a framework for the easy parts (prompt formatting, state passing)?

3. **The entire orchestrator is ~300–500 lines.** Here's the core loop:
   ```python
   for round in range(max_rounds):
       for agent in [us_agent, china_agent, europe_judge]:
           swap_model(agent.model_name)          # TabbyAPI load/unload
           response = generate(agent.prompt(state)) # OpenAI-compatible call
           state.add_turn(agent.name, response)
       if judge_says_done(state):
           break
   ```
   This is clearer, more debuggable, and more maintainable than any framework graph.

4. **Zero dependency risk.** CrewAI has had breaking API changes between versions. AutoGen split into two competing forks (AG2 vs AutoGen 0.4). LangGraph inherits the entire LangChain dependency tree. Your custom code depends on `requests` and nothing else.

### Runner-Up: LangGraph

If you want a framework, LangGraph is the best choice. Its graph-based state machine maps naturally to debate rounds, and you can insert model-management nodes between agent nodes. The built-in checkpointing lets you pause and resume debates across sessions. The cost is a steep learning curve and the LangChain dependency chain.

### Avoid: CrewAI for this use case

CrewAI is task-oriented (do A, then B, then C, done). Your system is conversation-oriented (A and B debate iteratively while C judges). You'd be fighting the framework's paradigm the entire time.

---

## 5. Performance Optimizations

### 5A. RAM Pre-loading (Double-Buffering)

**Current approach:** Load model from SSD → GPU → generate → unload → repeat. Each swap costs 5–15 seconds of disk I/O.

**Proposed approach:** Keep all 3 models loaded in system RAM (32 GB is plenty). Swap RAM → GPU only.

```
Timeline (current):
  [=====DISK→GPU 8s=====][====Generate 30s====][==Unload 1s==][=====DISK→GPU 10s=====]...

Timeline (proposed):
  [=RAM→GPU 2s=][====Generate 30s====][==Swap 2s==][====Generate 30s====]...
```

**Impact:**
- Model swap time drops from **5–15 seconds** to **1–3 seconds** (PCIe 3.0 x16 transfer).
- With 32 GB RAM, all three models (~6–9 GB each at 4-bit) fit in RAM simultaneously with room to spare.
- Total time saved per full debate round (3 swaps): **~15–35 seconds**.

**Implementation:** Pre-load all models into RAM at startup. When swapping, transfer weights from RAM to GPU rather than reading from disk. Both ExLlamaV2 and llama.cpp support this pattern.

### 5B. KV Cache Quantization

**What:** Quantize the Key-Value cache from FP16 to Q4 or Q8. llama.cpp supports `--cache-type-k q4_0 --cache-type-v q4_0`.

**Impact:** Reduces KV cache memory by **2–4×**. For a 14B model at 4K context, this saves ~1–2 GB of VRAM — critical headroom when your largest model already takes ~9 GB.

**Quality cost:** Minimal. Q8 KV cache is virtually lossless. Q4 shows minor perplexity increase but is acceptable for debate-style generation.

### 5C. Speculative Decoding (Optional)

Pair each debate model with a tiny ~0.5–1B "draft" model. The draft model predicts multiple tokens, and the main model verifies them in a single forward pass.

- **Expected speedup:** 1.3–1.8× for debate-style text (lower acceptance rate than code/factual text).
- **VRAM cost:** ~0.5–1 GB for the draft model.
- **Verdict:** Worth experimenting with, but not a priority. The RAM pre-loading and EXL2 speed gains are more impactful.

---

## 6. Context Management

### Current: Implicit (full transcript passed each time)

### Proposed: Tiered Sliding Window + Hierarchical Summarization

The current design has no explicit context strategy. As debates grow, transcripts will overflow model context windows. Here's a structured solution:

### Tiered Context Window (per agent turn)

| Tier | Content | Token Budget | Persistence |
|---|---|---|---|
| **1 — Persona** | System prompt + character sheet + debate rules | ~300–500 tokens | Always included |
| **2 — Debate State** | Running document: key claims per side, points of contention, evidence cited | ~200–400 tokens | Updated every round |
| **3 — Recent History** | Last 2–3 full verbatim turns | ~500–1500 tokens | Sliding window |
| **4 — Compressed Past** | Progressive summary of older turns | ~300–800 tokens | Hierarchically compressed |

**Total per turn: ~1,500–3,000 tokens** — fits within any model's context with ample room for generation.

### Hierarchical Summarization

```
Turns 1-3 → Round 1 Summary (~100-150 tokens)
Turns 4-6 → Round 2 Summary (~100-150 tokens)
Turns 7-9 → Round 3 Summary (~100-150 tokens)

Round 1-3 Summaries → Phase 1 Summary (~100-150 tokens)
```

This keeps memory usage **O(log n)** relative to debate length rather than O(n). A 20-round debate uses roughly the same context budget as a 5-round debate.

### For Very Long Debates (10+ rounds): Add RAG

Store all turns in a local vector database (ChromaDB or FAISS). When generating a turn, retrieve the top-K most relevant past turns via embedding similarity. The embedding model (e.g., `all-MiniLM-L6-v2`, ~80 MB) runs on CPU with zero VRAM cost.

---

## 7. Prompt Engineering

### 7A. Persona Anchoring System

Since you're hot-swapping models, **each invocation is stateless**. The model has no memory of previous turns beyond what you inject in the prompt. Every prompt must fully re-establish the persona.

**Recommended persona template:**
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

This is the single biggest quality risk in multi-agent debate systems. LLMs are trained to be helpful and agreeable, which fights against sustained disagreement.

**Countermeasures (use all of these):**

1. **Adversarial instructions:** "Your PRIMARY goal is to identify the weakest point in your opponent's last argument and attack it."

2. **Disagreement requirements:** "Your response MUST contain at least one point of explicit disagreement with the previous speaker."

3. **Structural conflict:** Assign ideologically opposed frameworks (realist vs. liberal internationalist, state-capitalism vs. free-market) that make agreement structurally impossible.

4. **Temperature settings:** Use **0.8–1.0** for debaters (creative, diverse arguments) and **0.3–0.5** for the judge (analytical, precise synthesis).

5. **Periodic devil's advocate injection:** Every 3–4 turns, add a meta-instruction: "The debate is converging. Introduce a controversial counterpoint that challenges the emerging consensus."

### 7C. Chain-of-Thought with Hidden Reasoning

Use a two-section output format for debater agents:
```
<thinking>
[Internal strategic reasoning — hidden from other agents]
</thinking>

<argument>
[The actual debate turn — passed to other agents and the judge]
</argument>
```

Only the `<argument>` block enters the shared transcript. The `<thinking>` block creates richer arguments (CoT improves reasoning quality) without cluttering the debate history.

### 7D. Judge Effectiveness

Make the judge use a **structured rubric** before delivering its verdict:
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

## 8. Web Research

### Current: DuckDuckGo Search API or SearxNG

### Proposed: DuckDuckGo + Trafilatura (primary) · Tavily (fallback)

| Component | Tool | Purpose | Privacy | Cost |
|---|---|---|---|---|
| **Search** | DuckDuckGo (`duckduckgo-search`) | Find relevant URLs | Good (no API key) | Free |
| **Extraction** | Trafilatura | Extract clean text from URLs | Best (fully local) | Free |
| **Fallback search** | Tavily (free tier) | When DDG rate-limits | Low (cloud API) | Free (1,000/mo) |

**Why this stack:**

1. **DuckDuckGo + Trafilatura = zero-cost, no-API-key, privacy-respecting pipeline.** DDG finds URLs; Trafilatura extracts the content locally and strips ads/navigation/boilerplate. The agent gets clean text ready for analysis.

2. **Trafilatura is the missing piece** in your current design. DuckDuckGo search returns snippets, not full articles. Without content extraction, agents can only reference search snippets — not deeply analyse source material. Trafilatura bridges this gap with zero cloud dependency.

3. **Tavily as fallback** provides AI-optimized results (pre-extracted content, relevance-ranked) for the 1,000 searches/month free tier. Use it when DDG rate-limits during heavy research phases.

4. **SearxNG remains viable** if you want maximum control. It requires Docker setup and maintenance, but aggregates 70+ search engines with no rate limits. Recommend it only if you find DDG's rate limits too restrictive.

### New Addition: Jina Reader for JS-Heavy Pages

Some modern web pages require JavaScript rendering that Trafilatura can't handle. For these, **Jina Reader** (`https://r.jina.ai/{url}`) returns clean markdown from any URL, including JS-rendered pages. Use it as a secondary extraction tool when Trafilatura returns empty content.

---

## 9. TTS / Podcast Production

### Current: XTTSv2 or Kokoro-82M

### Proposed: Fish Speech (primary) · Dia for dialogue post-processing

| TTS Engine | Quality | Voice Cloning | Multi-Speaker | VRAM | Speed | License |
|---|---|---|---|---|---|---|
| XTTSv2 | Very Good | Excellent (6s ref) | Yes | ~2–4 GB | 0.5–1× RT | Restrictive |
| Kokoro-82M | Good+ | **No** | Presets only | <0.5 GB | 5–10× RT | Apache 2.0 |
| **Fish Speech** | **Very Good** | **Yes (zero-shot)** | **Yes** | **~2–4 GB** | **1–3× RT** | **Apache 2.0** |
| Dia (Nari Labs) | Very Good | Limited | Native 2-speaker | ~4–6 GB | 0.5–1× RT | Apache 2.0 |
| StyleTTS2 | Excellent | Yes (style ref) | Yes | ~2–4 GB | 2–5× RT | MIT |
| Orpheus TTS | Excellent | Limited | Presets | ~4–6 GB | 0.5–1.5× RT | Apache 2.0 |

### Why Fish Speech Over XTTSv2

1. **Actively maintained.** Coqui (the company behind XTTSv2) shut down in late 2023. Fish Speech is actively developed with regular releases (v1.5+ as of 2025).

2. **Better license.** Fish Speech is Apache 2.0 (fully permissive). XTTSv2 has restrictive licensing that requires a commercial agreement for commercial use.

3. **Faster generation.** 1–3× real-time vs XTTSv2's 0.5–1× real-time. For a long podcast transcript, this difference is significant.

4. **Comparable quality** with strong zero-shot voice cloning from ~10–30 second reference clips.

5. **Better multilingual support** (Chinese, English, Japanese, and more) — useful if your agents occasionally quote sources in other languages.

### Why Not Kokoro-82M Alone

Kokoro is extremely fast and lightweight, but it **lacks voice cloning**. For a convincing 3-speaker podcast, you need distinct, consistent voices. Kokoro's built-in presets may not provide enough differentiation for a professional-sounding output.

### Dia as a Complementary Tool

Dia (Nari Labs, 1.6B parameters) is purpose-built for multi-speaker dialogue. It natively handles turn-taking, interruptions, and natural conversational patterns using `[S1]` / `[S2]` speaker tags. It even supports non-verbal cues like laughter.

**Limitation:** Dia natively supports only 2 speakers. For your 3-speaker podcast, consider:
- Use **Fish Speech** for the full 3-speaker generation (primary recommendation), OR
- Use **Dia** for 2-speaker dialogue segments (e.g., US vs China exchanges), then **Fish Speech** for the EU analyst narration/verdict sections.

### Other Notable Mentions

- **Orpheus TTS** (3B, Llama-based): Best emotional expressiveness (`<laugh>`, `<sigh>`, `<gasp>` tags). Consider if you want a more dramatic podcast style.
- **StyleTTS2**: Highest naturalness benchmark scores. More complex setup but MIT licensed. Good if quality is the absolute priority.

---

## 10. Quality & Evaluation

### New Addition: Automated Debate Quality Scoring

The current architecture has no evaluation mechanism. Add these layers:

### 10A. LLM-as-Judge (After Each Round)

After each debate round, run a quick evaluation pass using the judge model:
```
Evaluate the last round of debate. Score each participant on:
- Argument novelty (1-5): Did they introduce new points or repeat old ones?
- Evidence quality (1-5): Did they cite specific data, events, or sources?
- Engagement (1-5): Did they address the opponent's arguments directly?
- Logical coherence (1-5): Were there any fallacies or contradictions?
```

Use this to detect when the debate is stagnating (scores plateauing) and trigger the termination condition rather than relying solely on a time-based cutoff.

### 10B. Repetition Detection (Embedding Similarity)

After each turn, compute the semantic embedding (using `all-MiniLM-L6-v2`, ~80 MB, CPU-only) and compare against all previous turns from the same agent.

- **Cosine similarity > 0.85** with any prior turn → flag as repetitive
- **3 consecutive flagged turns** → end the debate early (agents have exhausted their arguments)

This runs on CPU with zero VRAM cost and prevents the debate from going in circles.

### 10C. Claim Tracking

Maintain a structured `debate_state` document that tracks:
```json
{
  "us_claims": ["claim1", "claim2"],
  "china_claims": ["claim1", "claim2"],
  "points_of_agreement": ["..."],
  "points_of_contention": ["..."],
  "evidence_cited": {"us": [...], "china": [...]}
}
```

Update this after each turn (the model can do this as part of its generation). This serves double duty: it feeds the context management system (Section 6) and provides a structured summary for the final verdict.

---

## 11. Summary: Before vs After

| Component | Current Design | Proposed Improvement | Key Benefit |
|---|---|---|---|
| **Inference Engine** | Ollama | ExLlamaV2 + TabbyAPI | **2× faster generation**, API-driven model management |
| **US Model** | Gemma 2 9B (8K ctx) | Gemma 3 12B (128K ctx) | **16× more context**, better reasoning |
| **China Model** | DeepSeek-R1-Distill-Qwen-14B | Same (keep) | Already optimal |
| **EU Judge Model** | Mistral Nemo 12B (MMLU ~73%) | Qwen 2.5 14B (MMLU ~80%) | **+7 MMLU points**, much stronger synthesis |
| **Quantization** | GGUF Q4_K_M (~4.8 bpw) | EXL2 variable (~4.0 bpw) | Better quality per bit, custom sizing |
| **Framework** | CrewAI / LangGraph | Custom Python (~400 lines) | Zero dependencies, full hot-swap control |
| **Model Swap Time** | 5–15s (disk→GPU) | 1–3s (RAM→GPU) | **5–10× faster swaps** via RAM pre-loading |
| **Context Strategy** | None (full transcript) | Tiered sliding window + summarization | O(log n) memory, never overflows |
| **Prompt Engineering** | Basic personas | Adversarial persona system + hidden CoT | Prevents agreement collapse, richer arguments |
| **Web Search** | DuckDuckGo only | DDG + Trafilatura + Tavily fallback | Full article extraction, not just snippets |
| **TTS Engine** | XTTSv2 (unmaintained) | Fish Speech (active, Apache 2.0) | Actively maintained, faster, permissive license |
| **Quality Control** | None | LLM-as-judge + embedding similarity | Detects stagnation, prevents circular debates |

---

## Implementation Priority

| Priority | Change | Effort | Impact |
|---|---|---|---|
| **P0 — Do First** | Upgrade models (Gemma 3 12B, Qwen 2.5 14B) | Low (download new models) | High |
| **P0 — Do First** | Switch to ExLlamaV2 + TabbyAPI | Medium (install + config) | High |
| **P1 — Core** | Build custom Python orchestrator | Medium (~400 lines) | High |
| **P1 — Core** | Implement tiered context management | Medium | High |
| **P1 — Core** | Implement adversarial prompt system | Low (prompt templates) | High |
| **P2 — Enhance** | RAM pre-loading for all 3 models | Low (config change) | Medium |
| **P2 — Enhance** | Add Trafilatura for web content extraction | Low (pip install) | Medium |
| **P2 — Enhance** | Switch TTS to Fish Speech | Medium (install + voice setup) | Medium |
| **P3 — Polish** | Add debate quality scoring | Medium | Medium |
| **P3 — Polish** | Add embedding-based repetition detection | Low | Low |
| **P3 — Polish** | Experiment with speculative decoding | High | Low-Medium |

---

*Generated by deep analysis of the current AI Swarm architecture against the state of the art in local LLM inference, multi-agent systems, quantization, TTS, and web search tooling as of March 2026.*
