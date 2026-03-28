# Politics AI Swarm

A fully local, multi-agent geopolitical debate system. Three specialized LLMs represent the US, China, and the EU in structured adversarial debates. Includes a real-time web dashboard with quality scoring, claim tracking, and podcast generation.

**No cloud dependencies. No API costs. Runs entirely on your machine.**

---

## How it works

Three LLM agents debate a geopolitical topic across multiple rounds:

| Agent | Model | bpw | Role |
|---|---|---|---|
| US Delegation | Gemma 3 12B IT (EXL2) | 4.0 | Debater -- US perspective |
| China Delegation | DeepSeek-R1-Distill-Qwen-14B (EXL2) | 3.5 | Debater -- Chinese perspective |
| EU Judge | Mistral Nemo Instruct 12B (EXL2) | 4.0 | Synthesizer / Evaluator |

The EU judge uses Mistral Nemo -- a European-developed model -- to avoid potential bias from Chinese or American models in the evaluation role.

Since only one model fits in VRAM at a time, the orchestrator hot-swaps models via TabbyAPI's SSE streaming API (RAM to GPU in 1-3 seconds).

### Debate flow

```
Phase 1 -- Research      Each agent searches the web and stakes an opening position
Phase 2 -- Debate loop   US argues -> China argues -> EU argues -> score -> repeat
Phase 3 -- Verdict       EU Judge delivers a structured ruling
Phase 4 -- Podcast       Transcript -> dialogue script -> Edge TTS -> .mp3 (optional)
```

### Anti-collapse system

LLMs tend toward agreement. Several mechanisms prevent this:
- Adversarial persona instructions requiring explicit disagreement each turn
- Hidden chain-of-thought (`<thinking>` blocks stripped from the debate log)
- Embedding-based repetition detection -- ends debate early if arguments stagnate
- Devil's advocate injection every N rounds when convergence is detected
- Quality scoring (novelty, evidence, engagement, coherence) with stagnation exit
- Distinct temperature settings (0.9 for debaters, 0.4 for the judge)

### Web dashboard

Real-time dashboard at `http://localhost:8000` with:
- **Live debate view** -- turns appear as they're generated, with research sources
- **Quality scores chart** -- per-round scoring for US and China
- **Claim tracker** -- side-by-side US vs China claims
- **Transcript browser** -- view past debates with full detail
- **Podcast player** -- YouTube-style synced transcript with audio playback
- Works for both web-launched and CLI-launched debates

### Script system

Pre-configured debate scenarios in `scripts/`:
```bash
python menu.py --script scripts/taiwan_crisis.yaml
```

Scripts define topic, rounds, personas, research toggle, and podcast output.

---

## Hardware requirements

| Component | Minimum |
|---|---|
| GPU | NVIDIA RTX 3060 (12 GB VRAM) |
| System RAM | 32 GB (holds all 3 models simultaneously) |
| Disk | ~25 GB free (models) |
| OS | Linux (tested on Ubuntu with CUDA) |

---

## Installation

**1. Clone the repo**

```bash
git clone --recurse-submodules <repo-url>
cd "Politics AI Swarm"
```

**2. Create the virtual environment and install dependencies**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install edge-tts
```

**3. Download models**

Models are downloaded from HuggingFace into `tabbyAPI/models/`:

```bash
source .venv/bin/activate
python -c "
from huggingface_hub import snapshot_download
snapshot_download('turboderp/gemma-3-12b-it-exl2', revision='4.0bpw',
                  local_dir='tabbyAPI/models/gemma-3-12b-it-exl2-4.0bpw')
snapshot_download('bartowski/DeepSeek-R1-Distill-Qwen-14B-exl2', revision='3_5',
                  local_dir='tabbyAPI/models/deepseek-r1-distill-qwen-14b-exl2-3.5bpw')
snapshot_download('turboderp/Mistral-Nemo-Instruct-12B-exl2', revision='4.0bpw',
                  local_dir='tabbyAPI/models/mistral-nemo-instruct-12b-exl2-4.0bpw')
"
```

> **Note:** Gemma 3 12B requires patching `config.json` to add `"max_position_embeddings": 8192` at the top level and inside `text_config` (TabbyAPI needs this for RoPE scaling).

**4. Start**

```bash
python menu.py
```

The menu auto-activates the `.venv` and starts TabbyAPI if it's not running.

---

## Usage

### Interactive menu

```bash
python menu.py
```

Options:
- **[1] Full run** -- research, debate, verdict, podcast
- **[2] Debate only** -- skip web research
- **[3] Quick test** -- 1 round, no research
- **[4] Run script** -- load a pre-configured scenario
- **[5] List scripts** -- browse available scripts
- **[s] System status** -- check TabbyAPI, models, transcripts

### Web dashboard

```bash
python dashboard.py          # http://localhost:8000
python dashboard.py --port 9000
```

### Run a script directly

```bash
python menu.py --script scripts/taiwan_crisis.yaml
```

### Script format

```yaml
topic: "Should there be a binding international treaty regulating frontier AI?"
rounds: 8
time_limit_minutes: 20
research: true
podcast: true
personas:
  us: lobbyist          # Director James Harrington
  china: economist      # Professor Zhang Yifei
  judge: economist      # Dr. Marie Leclerc
```

Available personas:
- **US:** `strategist` (default), `lobbyist`, `analyst`
- **China:** `director` (default), `enterprise_rep`, `economist`
- **EU:** `strategist` (default), `hawk`, `economist`

---

## Configuration

**`config/settings.yaml`** -- System settings:
- TabbyAPI URL and auth
- Model names, paths, generation params (temperature, top_p, max_tokens)
- Debate settings (rounds, time limit, repetition threshold)
- Research settings (DDG, Tavily API key, Jina)

**`config/personas.yaml`** -- Agent identities, beliefs, debate styles

**`tabbyAPI/config.yml`** -- TabbyAPI server config (`model_dir: models`)

---

## Podcast

Podcasts use **Edge TTS** (Microsoft's free neural voice API). No GPU needed, no API key.

Default voices:
- US: `en-US-GuyNeural` (male, authoritative)
- China: `en-US-AndrewNeural` (male, different tone)
- EU: `en-GB-SoniaNeural` (female, British)

Override voices from the web dashboard or by passing voice names in the script config.

The podcast pipeline:
1. LLM converts the debate transcript into a conversational dialogue script
2. Edge TTS generates MP3 audio for each segment
3. Segments are concatenated with a timing manifest for synced playback

---

## Project structure

```
.
├── menu.py                        # Interactive CLI menu (auto-activates .venv)
├── dashboard.py                   # Flask web dashboard
├── static/
│   └── index.html                 # Dashboard frontend
├── config/
│   ├── settings.yaml              # System configuration
│   └── personas.yaml              # Agent persona definitions
├── scripts/                       # Pre-configured debate scenarios
│   ├── lunar_base.yaml
│   ├── taiwan_crisis.yaml
│   ├── ai_regulation.yaml
│   └── belt_and_road.yaml
├── src/
│   ├── orchestrator.py            # Main 4-phase debate loop
│   ├── tabby_client.py            # TabbyAPI HTTP client (SSE model loading)
│   ├── live_status.py             # File-based cross-process debate tracking
│   ├── agents/
│   │   ├── base_agent.py          # Shared agent logic + tag parsing
│   │   ├── us_agent.py            # US Delegation
│   │   ├── china_agent.py         # China Delegation
│   │   └── eu_judge.py            # EU Judge
│   ├── context/
│   │   ├── context_manager.py     # Tiered context assembly
│   │   └── debate_state.py        # State tracker + serialization
│   ├── prompts/
│   │   └── templates.py           # All prompt templates
│   ├── research/
│   │   └── web_search.py          # DDG + Trafilatura + Jina + Tavily
│   ├── evaluation/
│   │   ├── quality_scorer.py      # LLM-as-judge scoring (novelty/evidence/engagement/coherence)
│   │   └── repetition_detector.py # Embedding similarity + early exit
│   ├── rag/
│   │   └── retriever.py           # ChromaDB RAG (10+ round debates)
│   └── tts/
│       └── podcast.py             # Edge TTS podcast pipeline
├── tabbyAPI/                      # Git submodule -- inference server
│   ├── config.yml                 # Pre-tuned for this project
│   └── models/                    # EXL2 model directories
├── output/
│   ├── transcripts/               # Debate transcripts (.json + .md)
│   └── audio/                     # Podcast audio (.mp3) + manifests
└── requirements.txt
```

---

## Tech stack

| Component | Technology |
|---|---|
| Inference | ExLlamaV2 + TabbyAPI |
| Quantization | EXL2 variable bitrate (3.5-4.0 bpw) |
| Orchestration | Custom Python debate loop |
| Web search | DuckDuckGo (`ddgs`) + Trafilatura + Jina Reader + Tavily |
| Embeddings / RAG | sentence-transformers (all-MiniLM-L6-v2, CPU) + ChromaDB |
| TTS | Edge TTS (Microsoft free neural voices) |
| Quality scoring | LLM-as-judge (novelty, evidence, engagement, coherence) |
| Repetition detection | Cosine similarity on sentence embeddings |
| Dashboard | Flask + vanilla JS with real-time polling |
| KV cache | Q8 quantization (saves ~1-2 GB VRAM) |
