"""
Quality scorer — uses the judge model (via TabbyAPI) to score each debate round.

Scores on 4 dimensions (1-5 each):
  - Argument novelty
  - Evidence quality
  - Engagement with opponent
  - Logical coherence

Returns per-agent scores dict. Stagnation: if average score drops below
threshold for N consecutive rounds, triggers early termination.
"""

import json
import re


SCORE_PROMPT = """\
Evaluate the following debate round. Score all three participants.

For each debater, score 1-5 on:
- novelty: Did they introduce new points not previously made?
- evidence: Did they cite specific data, events, or named sources?
- engagement: Did they directly address the opponents' specific arguments?
- coherence: Were the arguments logically consistent, no internal contradictions?

Output ONLY valid JSON in this exact format:
{{
  "us": {{"novelty": N, "evidence": N, "engagement": N, "coherence": N}},
  "china": {{"novelty": N, "evidence": N, "engagement": N, "coherence": N}},
  "judge": {{"novelty": N, "evidence": N, "engagement": N, "coherence": N}}
}}

ROUND TO EVALUATE:
{turns_text}

JSON:""".strip()


class QualityScorer:
    def __init__(self, stagnation_threshold: float = 2.5, stagnation_rounds: int = 3):
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_rounds = stagnation_rounds
        self._round_scores: list[dict] = []

    def score_round(self, turns: list, tabby_client) -> dict:
        """
        Score the given turns (one round). Returns scores dict.
        Falls back to neutral scores if LLM call fails.
        """
        turns_text = "\n\n".join(
            f"[{t.agent.upper()}]: {t.content}" for t in turns
        )
        if not turns_text.strip():
            return {}

        prompt = SCORE_PROMPT.format(turns_text=turns_text)
        messages = [{"role": "user", "content": prompt}]

        try:
            raw = tabby_client.chat(messages, temperature=0.2, max_tokens=400)
            # Extract JSON
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                scores = json.loads(raw[start:end])
                self._round_scores.append(scores)
                return scores
        except Exception as e:
            print(f"  [quality] scoring failed: {e}")

        neutral = {"novelty": 3, "evidence": 3, "engagement": 3, "coherence": 3}
        return {"us": neutral, "china": neutral, "judge": neutral}

    def should_terminate(self) -> bool:
        """True if average scores have been below threshold for stagnation_rounds."""
        if len(self._round_scores) < self.stagnation_rounds:
            return False

        recent = self._round_scores[-self.stagnation_rounds:]
        for round_scores in recent:
            for agent_scores in round_scores.values():
                avg = sum(agent_scores.values()) / len(agent_scores)
                if avg > self.stagnation_threshold:
                    return False  # At least one agent still producing quality
        return True

    def average_scores(self) -> dict:
        if not self._round_scores:
            return {}
        totals: dict[str, dict[str, float]] = {}
        for r in self._round_scores:
            for agent, scores in r.items():
                if agent not in totals:
                    totals[agent] = {k: 0.0 for k in scores}
                for k, v in scores.items():
                    totals[agent][k] += v
        n = len(self._round_scores)
        return {
            agent: {k: round(v / n, 2) for k, v in dims.items()}
            for agent, dims in totals.items()
        }
