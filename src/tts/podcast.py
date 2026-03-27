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

# Distinct voices for each speaker
DEFAULT_VOICES = {
    "us": "en-US-GuyNeural",            # US male, authoritative
    "china": "en-US-AndrewNeural",       # different male voice
    "judge": "en-GB-SoniaNeural",        # British female, European feel
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

        # Merge voice overrides
        voices = dict(DEFAULT_VOICES)
        if voice_refs:
            voices.update({k: v for k, v in voice_refs.items() if v})

        print(f"  [podcast] voices: US={voices['us']}, China={voices['china']}, EU={voices['judge']}")
        print(f"  [podcast] generating audio with Edge TTS...", flush=True)

        result = asyncio.run(
            self._generate_audio(script, voices, stamp, source_transcript)
        )
        return result or script_path

    def _generate_script(self, state: DebateState, tabby_client) -> list[dict]:
        prompt = SCRIPT_PROMPT.format(
            us_name=self.personas["us"]["personas"][self.personas["us"]["default_persona"]]["name"],
            china_name=self.personas["china"]["personas"][self.personas["china"]["default_persona"]]["name"],
            judge_name=self.personas["judge"]["personas"][self.personas["judge"]["default_persona"]]["name"],
            transcript=state.to_markdown(),
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            raw = tabby_client.chat(messages, temperature=0.7, max_tokens=4096)
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
                comm = edge_tts.Communicate(text, voice)
                await comm.save(str(out_path))

                if out_path.exists() and out_path.stat().st_size > 0:
                    duration = self._get_mp3_duration(out_path)
                    manifest_segments.append({
                        "index": i,
                        "speaker": speaker,
                        "agent": agent_key,
                        "text": text,
                        "start": round(current_time, 3),
                        "end": round(current_time + duration, 3),
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
        """Estimate MP3 duration from file size (128kbps average)."""
        size = path.stat().st_size
        # Edge TTS outputs ~128kbps MP3
        return size / (128 * 1000 / 8)
