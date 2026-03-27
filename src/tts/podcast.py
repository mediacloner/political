"""
Podcast production pipeline using Fish Speech.

Steps:
1. Convert debate transcript to 3-speaker dialogue script
2. Generate audio per turn using Fish Speech
3. Concatenate into final .wav

Fish Speech must be installed separately:
  git clone https://github.com/fishaudio/fish-speech
  pip install -e fish-speech/
"""

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


class PodcastProducer:
    def __init__(self, config: dict, personas: dict):
        self.config = config
        self.output_dir = Path(config["output"]["audio_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.personas = personas
        self._fish_available = None

    def produce(self, state: DebateState, tabby_client, voice_refs: dict = None,
                source_transcript: str = "") -> Optional[Path]:
        """
        Full pipeline: transcript → script → audio.
        voice_refs: {"us": "/path/to/ref.wav", "china": "...", "judge": "..."}
        Returns path to final .wav, or None if Fish Speech unavailable.
        """
        print("\n[podcast] generating dialogue script...", flush=True)
        script = self._generate_script(state, tabby_client)
        if not script:
            print("  [podcast] script generation failed")
            return None

        script_path = self.output_dir / f"script_{int(time.time())}.json"
        with open(script_path, "w") as f:
            json.dump(script, f, indent=2)
        print(f"  [podcast] script saved: {script_path}")

        if not self._check_fish_speech():
            print("  [podcast] Fish Speech not installed — script saved, audio skipped")
            print("  [podcast] install: git clone https://github.com/fishaudio/fish-speech && pip install -e fish-speech/")
            return script_path

        return self._generate_audio(script, voice_refs or {}, source_transcript)

    def _generate_script(self, state: DebateState, tabby_client) -> list[dict]:
        prompt = SCRIPT_PROMPT.format(
            us_name=self.personas["us"]["personas"][self.personas["us"]["default_persona"]]["name"],
            china_name=self.personas["china"]["personas"][self.personas["china"]["default_persona"]]["name"],
            judge_name=self.personas["judge"]["name"],
            transcript=state.to_markdown(),
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            raw = tabby_client.chat(messages, temperature=0.7, max_tokens=4096)
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start >= 0 and end > start:
                return json.loads(raw[start:end])
        except Exception as e:
            print(f"  [podcast] script generation error: {e}")
        return []

    def _check_fish_speech(self) -> bool:
        if self._fish_available is None:
            try:
                import fish_speech  # noqa
                self._fish_available = True
            except ImportError:
                self._fish_available = False
        return self._fish_available

    def _generate_audio(self, script: list[dict], voice_refs: dict,
                        source_transcript: str = "") -> Optional[Path]:
        """Generate audio per turn, record timing manifest, concatenate."""
        try:
            import subprocess
            import wave

            speaker_map = {
                "US_HOST": "us",
                "CHINA_GUEST": "china",
                "EU_ANALYST": "judge",
            }

            segment_paths = []
            manifest_segments = []
            current_time = 0.0

            for i, turn in enumerate(script):
                speaker = turn.get("speaker", "EU_ANALYST")
                agent_key = speaker_map.get(speaker, "judge")
                text = turn.get("text", "")
                ref_wav = voice_refs.get(agent_key, "")

                out_path = self.output_dir / f"seg_{i:04d}.wav"
                cmd = ["python", "-m", "fish_speech.cli", "tts",
                       "--text", text, "--output", str(out_path)]
                if ref_wav:
                    cmd += ["--reference-audio", ref_wav]

                result = subprocess.run(cmd, capture_output=True, timeout=60)
                if result.returncode == 0 and out_path.exists():
                    with wave.open(str(out_path), "rb") as wf:
                        duration = wf.getnframes() / wf.getframerate()
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
                else:
                    print(f"  [podcast] segment {i} failed: {result.stderr.decode()[:100]}")

            if not segment_paths:
                return None

            stem = int(time.time())
            final_path = self.output_dir / f"podcast_{stem}.wav"
            self._concatenate_wav(segment_paths, final_path)

            # Save timing manifest alongside the audio
            manifest = {
                "audio_file": final_path.name,
                "source_transcript": source_transcript,
                "total_duration": round(current_time, 3),
                "segments": manifest_segments,
            }
            manifest_path = self.output_dir / f"podcast_{stem}.manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

            # Clean up per-segment files
            for p in segment_paths:
                p.unlink(missing_ok=True)

            print(f"  [podcast] audio: {final_path}")
            print(f"  [podcast] manifest: {manifest_path}")
            return final_path

        except Exception as e:
            print(f"  [podcast] audio generation error: {e}")
            return None

    @staticmethod
    def _concatenate_wav(paths: list[Path], output: Path) -> None:
        import wave
        with wave.open(str(output), "wb") as out_wav:
            for i, p in enumerate(paths):
                with wave.open(str(p), "rb") as in_wav:
                    if i == 0:
                        out_wav.setparams(in_wav.getparams())
                    out_wav.writeframes(in_wav.readframes(in_wav.getnframes()))
