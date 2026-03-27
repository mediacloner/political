# Politics AI Swarm

A fully local, multi-agent geopolitical debate system. Three specialized LLMs represent the US, China, and the EU in structured adversarial debates. Output includes a full transcript and an optional multi-speaker audio podcast.

**No cloud dependencies. No API costs. Runs entirely on your machine.**

---

## How it works

Three LLM agents debate a geopolitical topic across multiple rounds:

| Agent | Model | Role |
|---|---|---|
| US Delegation | Gemma 3 12B (EXL2 4.5 bpw) | Debater — US perspective |
| China Delegation | DeepSeek-R1-Distill-Qwen-14B (EXL2 3.75 bpw) | Debater — Chinese perspective |
| EU Judge | Qwen 2.5 14B Instruct (EXL2 3.5 bpw) | Synthesiser / Evaluator |

Since only one model fits in VRAM at a time, the orchestrator hot-swaps models via TabbyAPI (RAM → GPU in 1–3 seconds). All three models are pre-loaded into system RAM at startup to eliminate disk I/O on every swap.

### Debate flow

```
Phase 1 — Research      Each agent searches the web and stakes an opening position
Phase 2 — Debate loop   Judge questions → US responds → China responds → score → repeat
Phase 3 — Verdict       Judge delivers a structured ruling (steelman → score → blind spots → verdict)
Phase 4 — Podcast       Transcript → dialogue script → Fish Speech TTS → .wav  (optional)
```

### Anti-collapse system

LLMs tend toward agreement. Several mechanisms prevent this:
- Adversarial persona instructions that require explicit disagreement each turn
- Hidden chain-of-thought (`<thinking>` blocks stripped from the debate log)
- Embedding-based repetition detection — ends debate early if arguments stagnate
- Devil's advocate injection every N rounds when convergence is detected
- Distinct temperature settings (0.9 for debaters, 0.4 for the judge)

---

## Hardware requirements

| Component | Minimum |
|---|---|
| GPU | NVIDIA RTX 3060 (12 GB VRAM) |
| System RAM | 32 GB (holds all 3 models simultaneously) |
| Disk | ~30 GB free (models) |
| OS | Linux (tested on Ubuntu with CUDA 13.x) |

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
pip install -e tabbyAPI/
pip install exllamav2
```

Or use the setup script (does all of the above):

```bash
bash setup_tabbyapi.sh
```

**3. Download models**

```bash
source activate.sh
python download_models.py --all
```

Models are downloaded from HuggingFace to `~/models/`. This requires ~25 GB of disk space and a HuggingFace account for some models.

To download individually:

```bash
python download_models.py --model gemma3     # US agent   (~7.5 GB)
python download_models.py --model deepseek   # China agent (~8.0 GB)
python download_models.py --model qwen       # EU judge    (~8.5 GB)
```

**4. Start TabbyAPI**

```bash
source activate.sh
cd tabbyAPI && python3 main.py
```

TabbyAPI will start on `http://localhost:5000`. The config is pre-tuned for this project at `tabbyAPI/config.yml`.

**5. Verify everything works**

```bash
source activate.sh
python main.py --check
# TabbyAPI is reachable at http://localhost:5000
```

---

## Usage

```bash
source activate.sh

# Basic debate
python main.py --topic "Should the ESA partner with NASA or CNSA for a lunar base?"

# More options
python main.py \
  --topic "Is Western financial sanctions policy effective against China?" \
  --rounds 6 \
  --time-limit 30 \
  --us-persona lobbyist \
  --china-persona economist

# With podcast output (requires Fish Speech — see below)
python main.py --topic "..." --podcast --voice-us voices/us.wav
```

### CLI reference

| Flag | Default | Description |
|---|---|---|
| `--topic` | required | Debate topic |
| `--rounds` | 8 | Maximum debate rounds |
| `--time-limit` | 20 | Time limit in minutes |
| `--us-persona` | `strategist` | `strategist` / `lobbyist` / `analyst` |
| `--china-persona` | `director` | `director` / `enterprise_rep` / `economist` |
| `--podcast` | off | Generate audio podcast after debate |
| `--voice-us` | none | Path to US voice reference clip (.wav, 10–30s) |
| `--voice-china` | none | Path to China voice reference clip |
| `--voice-judge` | none | Path to EU judge voice reference clip |
| `--config` | `config/settings.yaml` | Path to settings file |
| `--check` | — | Verify TabbyAPI is reachable and exit |

### Output

Transcripts are saved to `output/transcripts/` as both `.json` and `.md`. Audio (if enabled) goes to `output/audio/`.

---

## Configuration

Edit `config/settings.yaml` to tune:

- TabbyAPI URL and auth key
- Model names and paths
- Generation parameters (temperature, max tokens)
- Debate settings (rounds, time limit, window sizes)
- Web research settings (Tavily API key, result count)

Edit `config/personas.yaml` to modify agent identities, beliefs, and rhetorical styles.

---

## Podcast (optional)

Install Fish Speech:

```bash
git clone https://github.com/fishaudio/fish-speech
.venv/bin/pip install -e fish-speech/
```

Record 10–30 second voice reference clips for each speaker, then:

```bash
python main.py --topic "..." --podcast \
  --voice-us voices/us_ref.wav \
  --voice-china voices/china_ref.wav \
  --voice-judge voices/judge_ref.wav
```

Without voice references, Fish Speech uses its default voice for all speakers.

---

## Project structure

```
.
├── main.py                        # CLI entrypoint
├── download_models.py             # HuggingFace EXL2 model downloader
├── activate.sh                    # Activate virtualenv
├── config/
│   ├── settings.yaml              # System configuration
│   └── personas.yaml              # Agent persona definitions
├── src/
│   ├── orchestrator.py            # Main 4-phase debate loop
│   ├── tabby_client.py            # TabbyAPI HTTP client
│   ├── agents/
│   │   ├── base_agent.py          # Shared agent logic
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
│   │   ├── quality_scorer.py      # LLM-as-judge per round
│   │   └── repetition_detector.py # Embedding similarity + early exit
│   ├── rag/
│   │   └── retriever.py           # ChromaDB RAG (10+ round debates)
│   └── tts/
│       └── podcast.py             # Fish Speech podcast pipeline
├── tabbyAPI/                      # Git submodule — inference server
├── output/
│   ├── transcripts/               # Debate transcripts (.json + .md)
│   └── audio/                     # Podcast audio (.wav)
└── Master Architecture - Integrated.md  # Full design reference
```

---

## Design documents

- `Master Architecture - Integrated.md` — Complete architecture reference (models, stack, context management, prompt engineering)
- `AI Swarm - Improvements Report.md` — Detailed rationale for every technology choice
- `TASKS.md` — Implementation status and remaining setup steps

---

## Tech stack

| Component | Technology |
|---|---|
| Inference | ExLlamaV2 + TabbyAPI |
| Quantization | EXL2 variable bitrate (3.5–4.5 bpw) |
| Orchestration | Custom Python (~400 lines) |
| Web search | DuckDuckGo + Trafilatura + Jina Reader + Tavily |
| Embeddings / RAG | sentence-transformers (all-MiniLM-L6-v2, CPU) + ChromaDB |
| TTS | Fish Speech (Apache 2.0) |
| KV cache | Q8 quantization (saves ~1–2 GB VRAM) |
