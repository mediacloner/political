"""
DebateState — single source of truth for all debate data.
Tracks turns, claims, evidence, summaries, and quality scores.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Turn:
    agent: str          # "us" | "china" | "judge"
    round_num: int
    content: str        # The <argument> block (public)
    thinking: str = ""  # The <thinking> block (hidden)
    timestamp: float = field(default_factory=time.time)
    quality_score: Optional[dict] = None   # Set by quality scorer
    is_repetitive: bool = False            # Set by repetition detector


@dataclass
class DebateState:
    topic: str
    round_num: int = 0
    turns: list = field(default_factory=list)
    us_claims: list = field(default_factory=list)
    china_claims: list = field(default_factory=list)
    points_of_agreement: list = field(default_factory=list)
    points_of_contention: list = field(default_factory=list)
    evidence_cited: dict = field(default_factory=lambda: {"us": [], "china": []})
    summaries: dict = field(default_factory=dict)   # {"round_1": "...", "phase_1": "..."}
    verdict: str = ""
    finished: bool = False
    finish_reason: str = ""  # "rounds_exhausted" | "time_limit" | "stagnation" | "judge_done"

    # ------------------------------------------------------------------
    # Turn management
    # ------------------------------------------------------------------

    def add_turn(self, agent: str, content: str, thinking: str = "") -> Turn:
        turn = Turn(
            agent=agent,
            round_num=self.round_num,
            content=content,
            thinking=thinking,
        )
        self.turns.append(turn)
        return turn

    def get_turns_by_agent(self, agent: str) -> list:
        return [t for t in self.turns if t.agent == agent]

    def get_recent_turns(self, n: int = 3) -> list:
        """Return the last n turns (all agents combined)."""
        return self.turns[-n:] if len(self.turns) >= n else self.turns[:]

    def get_turns_for_round(self, round_num: int) -> list:
        return [t for t in self.turns if t.round_num == round_num]

    def get_unsummarized_turns(self) -> list:
        """Turns from rounds that don't yet have a summary."""
        summarized_rounds = {
            int(k.split("_")[1]) for k in self.summaries if k.startswith("round_")
        }
        return [t for t in self.turns if t.round_num not in summarized_rounds]

    # ------------------------------------------------------------------
    # Claim / evidence management
    # ------------------------------------------------------------------

    def add_claim(self, agent: str, claim: str) -> None:
        if agent == "us":
            if claim not in self.us_claims:
                self.us_claims.append(claim)
        elif agent == "china":
            if claim not in self.china_claims:
                self.china_claims.append(claim)

    def add_evidence(self, agent: str, source: str) -> None:
        if agent in self.evidence_cited and source not in self.evidence_cited[agent]:
            self.evidence_cited[agent].append(source)

    def add_summary(self, key: str, summary: str) -> None:
        self.summaries[key] = summary

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "round_num": self.round_num,
            "finished": self.finished,
            "finish_reason": self.finish_reason,
            "verdict": self.verdict,
            "us_claims": self.us_claims,
            "china_claims": self.china_claims,
            "points_of_agreement": self.points_of_agreement,
            "points_of_contention": self.points_of_contention,
            "evidence_cited": self.evidence_cited,
            "summaries": self.summaries,
            "turns": [
                {
                    "agent": t.agent,
                    "round": t.round_num,
                    "content": t.content,
                    "thinking": t.thinking,
                    "timestamp": t.timestamp,
                    "quality_score": t.quality_score,
                    "is_repetitive": t.is_repetitive,
                }
                for t in self.turns
            ],
        }

    def to_markdown(self) -> str:
        lines = [f"# Debate: {self.topic}\n"]
        lines.append(f"**Rounds completed:** {self.round_num}")
        lines.append(f"**Finish reason:** {self.finish_reason}\n")

        for turn in self.turns:
            agent_label = {"us": "🇺🇸 US Delegation", "china": "🇨🇳 China Delegation", "judge": "🇪🇺 EU Judge"}
            label = agent_label.get(turn.agent, turn.agent.upper())
            lines.append(f"## {label} — Round {turn.round_num}\n")
            lines.append(turn.content)
            lines.append("")

        if self.verdict:
            lines.append("---\n## 🇪🇺 Final Verdict\n")
            lines.append(self.verdict)

        return "\n".join(lines)

    def save(self, output_dir: str) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        stem = f"debate_{int(time.time())}"

        with open(path / f"{stem}.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        with open(path / f"{stem}.md", "w") as f:
            f.write(self.to_markdown())

        print(f"  [state] saved to {path / stem}.[json|md]")
