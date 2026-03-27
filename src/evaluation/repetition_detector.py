"""
Repetition detector — uses sentence-transformers (CPU) to compute
cosine similarity between debate turns.

If any prior turn from the same agent scores > threshold, the new turn
is flagged as repetitive. Three consecutive flags trigger early termination.
"""

from typing import Optional


class RepetitionDetector:
    def __init__(self, threshold: float = 0.85, max_consecutive: int = 3):
        self.threshold = threshold
        self.max_consecutive = max_consecutive
        self._model = None
        self._embeddings: dict[str, list] = {}  # agent -> list of embeddings
        self._consecutive_flags: dict[str, int] = {"us": 0, "china": 0}

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
                print("  [repetition] loaded all-MiniLM-L6-v2", flush=True)
            except ImportError:
                print("  [repetition] sentence-transformers not installed, skipping")
        return self._model

    def check(self, agent: str, text: str) -> tuple[bool, float]:
        """
        Returns (is_repetitive, max_similarity).
        Adds the embedding to history regardless of result.
        """
        model = self._get_model()
        if model is None:
            return False, 0.0

        import numpy as np

        import numpy as np

        embedding = model.encode([text])[0]

        prior = self._embeddings.get(agent, [])
        max_sim = 0.0

        if prior:
            prior_stack = np.stack(prior)
            similarities = model.similarity([embedding], prior_stack)[0]
            max_sim = float(similarities.max())

        # Store embedding
        if agent not in self._embeddings:
            self._embeddings[agent] = []
        self._embeddings[agent].append(embedding)

        is_repetitive = max_sim > self.threshold

        if agent in self._consecutive_flags:
            if is_repetitive:
                self._consecutive_flags[agent] += 1
            else:
                self._consecutive_flags[agent] = 0

        return is_repetitive, max_sim

    def should_terminate(self) -> bool:
        """True if any debater has hit max_consecutive repetitive turns."""
        return any(
            count >= self.max_consecutive
            for count in self._consecutive_flags.values()
        )

    def reset(self) -> None:
        self._embeddings.clear()
        self._consecutive_flags = {"us": 0, "china": 0}
