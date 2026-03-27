# Master Architecture Document: Geo-Distributed AI Swarm

## 1. Executive Summary
This project outlines the development of a fully local, multi-agent Mixture of Experts (MoE) swarm designed to simulate a geopolitical war room. The system leverages distinct open-weights Large Language Models (LLMs) representing the United States, China, and Europe. By employing sequential model loading ("hot-swapping"), the system facilitates complex, multi-persona debates, dynamic web research, and iterative cross-examination, culminating in a synthesised European verdict and an auto-generated, multi-speaker audio podcast.

## 2. Hardware Specifications & Constraints
The architecture is rigorously optimised for local execution to ensure absolute data privacy and zero API costs.

* **Target GPU:** Nvidia RTX 3060
* **Video RAM (VRAM):** 12GB (Dictates a sequential model execution paradigm)
* **System RAM:** 32GB (Serves as the memory buffer for rapid model swapping)
* **Execution Strategy:** **Sequential "Hot-Swapping".** Due to the 12GB VRAM constraint, only one model is loaded onto the GPU at any given time. Once an agent completes its turn, the inference engine purges it from memory before loading the next.

## 3. The Multi-Agent Neural Network
The swarm utilises a roster of highly capable, quantised (4-bit) models, explicitly chosen for their regional alignment, reasoning capabilities, and VRAM efficiency.

### Node A: The United States Delegation
* **Core Model:** `Gemma 2 9B` (Google)
* **VRAM Footprint:** ~6.5GB (4-bit quantisation)
* **Strengths:** Robust generalist reasoning, deep structural analysis, and strict instruction adherence. 
* **Dynamic Personas:** US Geopolitical Strategist, US Tech Corporate Lobbyist, US Financial Analyst.

### Node B: The Chinese Delegation
* **Core Model:** `DeepSeek-R1-Distill-Qwen-14B` (DeepSeek / Alibaba)
* **VRAM Footprint:** ~9.0GB (4-bit quantisation)
* **Strengths:** Unparalleled local mathematical reasoning, algorithmic efficiency, and highly structured, first-principles logic.
* **Dynamic Personas:** PRC Strategic Affairs Director, State-Owned Enterprise Representative, Macro-Economist.

### Node C: The European Synthesiser (The Judge)
* **Core Model:** `Mistral Nemo 12B` (Mistral AI)
* **VRAM Footprint:** ~8.0GB (4-bit quantisation)
* **Strengths:** Massive 128k context window, exceptional multi-turn synthesis, and objective harmonisation of conflicting data.
* **Dynamic Persona:** European Commission Lead Strategist.

## 4. Software Stack & Orchestration
* **Local Inference Engine:** **Ollama**. Used for model serving. The `keep_alive=0` parameter is strictly enforced to ensure immediate VRAM purging after each generation.
* **Multi-Agent Framework:** **CrewAI** or **LangGraph**. Responsible for managing the conversational state, routing prompts to the correct model, and multiplexing the personas.
* **Web Research Tool:** **DuckDuckGo Search API** (or local SearxNG). Allows agents to pull real-time data to substantiate their arguments.
* **Audio Synthesis:** **XTTSv2** or **Kokoro-82M**. Lightweight, local Text-to-Speech (TTS) engines for generating the final multi-speaker podcast.

## 5. Case Study: The Space Race Summit
To illustrate the system's operational flow, consider the following simulation.

**The Prompt:** *"Should the European Space Agency (ESA) partner with NASA's Artemis programme or the CNSA's International Lunar Research Station (ILRS) for the establishment of a permanent lunar base?"*

### Phase 1: Research and Initial Proposals
1. The US Model (`Gemma 2`) is loaded. Its system prompt is set to the *US Tech Corporate Lobbyist*. It searches the web for current Artemis contracts and generates a proposal highlighting commercial aerospace benefits. The model is unloaded.
2. The Chinese Model (`DeepSeek-R1-14B`) is loaded as the *PRC Strategic Affairs Director*. It researches the Chang'e timeline and drafts a proposal focused on resource efficiency and guaranteed timelines. It is unloaded.

### Phase 2: The Iterative Debate Loop (Time-Bound)
The system enters a `while` loop, bounded by a user-defined time limit (e.g., 20 minutes).
1. **The Inquisition:** The European Judge (`Mistral Nemo`) reviews the transcript and identifies a financial discrepancy. It generates a targeted question: *"US Analyst, how do you mitigate the risk of congressional budget cuts delaying the Artemis timeline compared to China's state-backed funding?"*
2. **The Rebuttal:** The US Model is reloaded, researches recent congressional space budgets, and provides a counter-argument.
3. *This cross-examination loops until the timer hits the 18-minute mark.*

### Phase 3: The Verdict & Transcript Generation
1. The debate is halted.
2. The European Judge (`Mistral Nemo`) ingests the entire debate transcript. It weighs the geopolitical, financial, and technical arguments, outputting a definitive ruling on whether the ESA should side with the US or China.
3. The complete textual transcript is saved to the local drive.

### Phase 4: Podcast Production
1. A lightweight summarisation script converts the raw transcript into a natural dialogue script featuring three speakers (US Host, Chinese Guest, EU Analyst).
2. All LLMs are purged from VRAM.
3. The TTS engine (`XTTSv2`) is loaded. It assigns distinct voice clones to each speaker, reads the script sequentially, and compiles the audio into a final `.wav` file.