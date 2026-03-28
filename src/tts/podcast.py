"""
Podcast production pipeline using Edge TTS (Microsoft free neural voices).

Steps:
1. Convert debate transcript to 3-speaker dialogue script (via LLM)
2. Generate audio per segment using Edge TTS
3. Concatenate into final .mp3 with timing manifest
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Optional
from src.context.debate_state import DebateState


SCRIPT_PROMPT = """\
Convert the following geopolitical debate transcript into a natural-sounding
podcast dialogue script. The podcast has three hosts:
- "US_HOST": {us_name} (US perspective)
- "CHINA_GUEST": {china_name} (Chinese perspective)
- "EU_ANALYST": {judge_name} (European analyst and host)

Rules:
- Keep the substance of each argument but make it conversational
- EU_ANALYST introduces topics, asks follow-up questions, and delivers the closing analysis
- Each speaking turn should be 2-5 sentences
- Add natural transitions ("That's an interesting point", "Let me push back on that...")
- Output as JSON array: [{{"speaker": "US_HOST", "text": "..."}}, ...]

TRANSCRIPT:
{transcript}

JSON:""".strip()

# Voice pairs per agent: (female_voice, male_voice)
VOICE_PAIRS = {
    "us":    ("en-US-JennyNeural",   "en-US-GuyNeural"),
    "china": ("en-US-AriaNeural",    "en-US-AndrewNeural"),
    "judge": ("en-GB-SoniaNeural",   "en-GB-RyanNeural"),
}

# Persona name → gender (derived from personas.yaml)
PERSONA_GENDER = {
    # US
    "Ambassador Victoria Marsh": "f",
    "Director James Harrington": "m",
    "Dr. Sarah Chen": "f",
    # China
    "Director-General Wei Changming": "m",
    "Chairman Liu Peng": "m",
    "Professor Zhang Yifei": "m",
    # Judge / EU
    "Commissioner Elise Fontaine": "f",
    "Admiral Henrik Sørensen": "m",
    "Dr. Marie Leclerc": "f",
}

# Fallback defaults (kept for backward compat with voice_refs override)
DEFAULT_VOICES = {
    "us": "en-US-JennyNeural",          # US female (default persona is Victoria Marsh)
    "china": "en-US-AndrewNeural",       # male
    "judge": "en-GB-SoniaNeural",        # British female
}

SPEAKER_MAP = {
    "US_HOST": "us",
    "CHINA_GUEST": "china",
    "EU_ANALYST": "judge",
}


class PodcastProducer:
    def __init__(self, config: dict, personas: dict):
        self.config = config
        self.output_dir = Path(config["output"]["audio_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.personas = personas

    def produce(self, state: DebateState, tabby_client, voice_refs: dict = None,
                source_transcript: str = "") -> Optional[Path]:
        """
        Full pipeline: transcript -> script -> audio.
        voice_refs: {"us": "en-US-GuyNeural", "china": "...", "judge": "..."}
                    Override voice names per speaker.
        Returns path to final audio, or script path if audio fails.
        """
        print("\n[podcast] generating dialogue script...", flush=True)
        script = self._generate_script(state, tabby_client)
        if not script:
            print("  [podcast] script generation failed")
            return None

        stamp = int(time.time())
        script_path = self.output_dir / f"script_{stamp}.json"
        with open(script_path, "w") as f:
            json.dump(script, f, indent=2)
        print(f"  [podcast] script saved: {script_path} ({len(script)} segments)")

        # Select voices based on persona gender, then apply any overrides
        voices = self._resolve_voices()
        if voice_refs:
            voices.update({k: v for k, v in voice_refs.items() if v})

        print(f"  [podcast] voices: US={voices['us']}, China={voices['china']}, EU={voices['judge']}")
        print(f"  [podcast] generating audio with Edge TTS...", flush=True)

        result = asyncio.run(
            self._generate_audio(script, voices, stamp, source_transcript)
        )
        return result or script_path

    def _resolve_voices(self) -> dict:
        """Pick male/female voice per agent based on the active persona's gender."""
        voices = {}
        for agent_key in ("us", "china", "judge"):
            agent_cfg = self.personas.get(agent_key, {})
            default_p = agent_cfg.get("default_persona", "")
            persona = agent_cfg.get("personas", {}).get(default_p, {})
            name = persona.get("name", "")
            gender = PERSONA_GENDER.get(name, "f" if agent_key == "judge" else "m")
            female_voice, male_voice = VOICE_PAIRS[agent_key]
            voices[agent_key] = female_voice if gender == "f" else male_voice
        return voices

    def _compact_transcript(self, state: DebateState, max_chars: int = 4000) -> str:
        """Build a compact transcript that fits in the context window."""
        lines = [f"# Debate: {state.topic}\n"]
        agent_label = {"us": "US", "china": "China", "judge": "EU Judge"}
        total = 0
        for turn in state.turns:
            label = agent_label.get(turn.agent, turn.agent)
            # Truncate long turns
            content = turn.content[:600] + "..." if len(turn.content) > 600 else turn.content
            entry = f"\n**{label} (Round {turn.round_num}):**\n{content}\n"
            if total + len(entry) > max_chars:
                lines.append("\n[...remaining turns truncated for brevity...]")
                break
            lines.append(entry)
            total += len(entry)
        if state.verdict:
            verdict = state.verdict[:800] + "..." if len(state.verdict) > 800 else state.verdict
            lines.append(f"\n**EU Judge -- Final Verdict:**\n{verdict}")
        return "\n".join(lines)

    def _generate_script(self, state: DebateState, tabby_client) -> list[dict]:
        transcript = self._compact_transcript(state)
        prompt = SCRIPT_PROMPT.format(
            us_name=self.personas["us"]["personas"][self.personas["us"]["default_persona"]]["name"],
            china_name=self.personas["china"]["personas"][self.personas["china"]["default_persona"]]["name"],
            judge_name=self.personas["judge"]["personas"][self.personas["judge"]["default_persona"]]["name"],
            transcript=transcript,
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            # Use longer timeout for script generation (large output)
            import requests
            payload = {
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2048,
                "stream": False,
            }
            resp = requests.post(
                f"{tabby_client.base_url}/v1/chat/completions",
                json=payload,
                headers=tabby_client.headers,
                timeout=300,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            # Strip any thinking tags
            import re
            cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
            start = cleaned.find("[")
            end = cleaned.rfind("]") + 1
            if start >= 0 and end > start:
                return json.loads(cleaned[start:end])
        except Exception as e:
            print(f"  [podcast] script generation error: {e}")
        return []

    async def _generate_audio(self, script: list[dict], voices: dict,
                               stamp: int, source_transcript: str) -> Optional[Path]:
        """Generate audio per segment using Edge TTS, concatenate."""
        try:
            import edge_tts
        except ImportError:
            print("  [podcast] edge-tts not installed: pip install edge-tts")
            return None

        segment_paths = []
        manifest_segments = []
        current_time = 0.0

        for i, turn in enumerate(script):
            speaker = turn.get("speaker", "EU_ANALYST")
            agent_key = SPEAKER_MAP.get(speaker, "judge")
            text = turn.get("text", "").strip()
            if not text:
                continue

            voice = voices.get(agent_key, DEFAULT_VOICES["judge"])
            out_path = self.output_dir / f"seg_{stamp}_{i:04d}.mp3"

            try:
                comm = edge_tts.Communicate(text, voice, boundary="WordBoundary")
                audio_data = b''
                word_boundaries = []
                async for chunk in comm.stream():
                    if chunk["type"] == "audio":
                        audio_data += chunk["data"]
                    elif chunk["type"] == "WordBoundary":
                        wb_start = chunk["offset"] / 10_000_000
                        wb_dur = chunk["duration"] / 10_000_000
                        word_boundaries.append({
                            "word": chunk["text"],
                            "start": round(current_time + wb_start, 3),
                            "end": round(current_time + wb_start + wb_dur, 3),
                        })

                if audio_data:
                    out_path.write_bytes(audio_data)
                    duration = self._get_mp3_duration(out_path)
                    manifest_segments.append({
                        "index": i,
                        "speaker": speaker,
                        "agent": agent_key,
                        "text": text,
                        "start": round(current_time, 3),
                        "end": round(current_time + duration, 3),
                        "words": word_boundaries,
                    })
                    current_time += duration
                    segment_paths.append(out_path)
                    print(f"  [podcast] seg {i+1}/{len(script)}: {speaker} ({duration:.1f}s)", flush=True)
                else:
                    print(f"  [podcast] seg {i} empty output")
            except Exception as e:
                print(f"  [podcast] seg {i} failed: {e}")

        if not segment_paths:
            print("  [podcast] no segments generated")
            return None

        # Concatenate MP3 segments (simple binary concat works for MP3)
        final_path = self.output_dir / f"podcast_{stamp}.mp3"
        with open(final_path, "wb") as out:
            for p in segment_paths:
                out.write(p.read_bytes())

        # Save manifest
        manifest = {
            "audio_file": final_path.name,
            "source_transcript": source_transcript,
            "total_duration": round(current_time, 3),
            "segments": manifest_segments,
        }
        manifest_path = self.output_dir / f"podcast_{stamp}.manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Clean up segments
        for p in segment_paths:
            p.unlink(missing_ok=True)

        print(f"  [podcast] audio saved: {final_path} ({current_time:.1f}s total)")
        print(f"  [podcast] manifest: {manifest_path}")
        return final_path

    @staticmethod
    def _get_mp3_duration(path: Path) -> float:
        """Get MP3 duration by parsing frame header for actual bitrate."""
        data = path.read_bytes()
        # MPEG1-Layer3 and MPEG2-Layer3 bitrate tables (kbps)
        v1l3 = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0]
        v2l3 = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0]
        for i in range(min(len(data) - 4, 4096)):
            if data[i] != 0xFF or (data[i + 1] & 0xE0) != 0xE0:
                continue
            ver = (data[i + 1] >> 3) & 0x3
            layer = (data[i + 1] >> 1) & 0x3
            br_idx = (data[i + 2] >> 4) & 0xF
            if br_idx == 0 or br_idx == 15 or ver == 1 or layer == 0:
                continue
            kbps = (v1l3 if ver == 3 else v2l3)[br_idx]
            if kbps > 0:
                return (len(data) - i) / (kbps * 125)
            break
        return len(data) / 6000  # fallback ~48kbps
