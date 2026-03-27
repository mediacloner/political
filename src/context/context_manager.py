"""
ContextManager — assembles the tiered context window for each agent turn.

Tier 1: Persona system prompt         (~300-500 tokens, always present)
Tier 2: Debate state summary          (~200-400 tokens)
Tier 3: Last N verbatim turns         (~500-1500 tokens, sliding window)
Tier 4: Hierarchical summaries        (~300-800 tokens, O(log n) growth)
"""

from src.context.debate_state import DebateState
from src.prompts.templates import (
    build_debate_state_block,
    build_recent_history_block,
    build_compressed_past_block,
    SUMMARIZE_ROUND,
    SUMMARIZE_PHASE,
)


class ContextManager:
    def __init__(self, config: dict, tabby_client=None):
        self.recent_turns_window = config["debate"]["recent_turns_window"]
        self.summary_every_n = config["debate"]["summary_every_n_turns"]
        self.tabby_client = tabby_client  # Used for summarization calls

    # ------------------------------------------------------------------
    # Public: build the user message for a debater turn
    # ------------------------------------------------------------------

    def build_debater_turn_messages(
        self,
        system_prompt: str,
        state: DebateState,
        turn_objective: str,
        extra_instruction: str = "",
    ) -> list[dict]:
        """
        Returns a messages list ready for chat completion:
        [{"role": "system", ...}, {"role": "user", ...}]
        """
        user_content = self._build_user_content(state, turn_objective, extra_instruction)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def build_judge_messages(
        self,
        system_prompt: str,
        state: DebateState,
        turn_objective: str,
    ) -> list[dict]:
        """Same structure but judge sees more history."""
        user_content = self._build_user_content(state, turn_objective, recent_n=6)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def build_verdict_messages(self, system_prompt: str, state: DebateState, verdict_prompt: str) -> list[dict]:
        """Judge verdict gets the full state context."""
        all_turns_text = self._format_all_turns(state)
        user_content = f"{all_turns_text}\n\n---\n{verdict_prompt}"
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    # ------------------------------------------------------------------
    # Summarization — call TabbyAPI with current judge model
    # ------------------------------------------------------------------

    def maybe_summarize(self, state: DebateState) -> None:
        """
        If a full set of N turns has accumulated without a summary, generate one.
        Also generate a phase summary every 3 round summaries.
        """
        if self.tabby_client is None:
            return

        rounds_without_summary = set()
        for turn in state.get_unsummarized_turns():
            rounds_without_summary.add(turn.round_num)

        for rnum in sorted(rounds_without_summary):
            round_turns = state.get_turns_for_round(rnum)
            if len(round_turns) < 2:
                continue  # Incomplete round, wait
            summary = self._summarize_round(rnum, round_turns)
            state.add_summary(f"round_{rnum}", summary)

        # Phase summaries: every 3 round summaries
        round_summaries = {k: v for k, v in state.summaries.items() if k.startswith("round_")}
        round_nums_summarized = sorted(int(k.split("_")[1]) for k in round_summaries)
        phase_size = 3

        for i in range(0, len(round_nums_summarized), phase_size):
            chunk = round_nums_summarized[i:i + phase_size]
            if len(chunk) < phase_size:
                break
            phase_key = f"phase_{i // phase_size + 1}"
            if phase_key not in state.summaries:
                chunk_summaries = {f"round_{r}": round_summaries[f"round_{r}"] for r in chunk}
                phase_summary = self._summarize_phase(chunk_summaries)
                state.add_summary(phase_key, phase_summary)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_user_content(
        self,
        state: DebateState,
        turn_objective: str,
        extra_instruction: str = "",
        recent_n: int = None,
    ) -> str:
        n = recent_n or self.recent_turns_window
        tier2 = build_debate_state_block(state)
        tier3 = build_recent_history_block(state.get_recent_turns(n))
        tier4 = build_compressed_past_block(
            {k: v for k, v in state.summaries.items()}
        )

        parts = []
        if tier2:
            parts.append(tier2)
        if tier4:
            parts.append(tier4)
        if tier3:
            parts.append(tier3)

        objective_block = f"---\nYOUR OBJECTIVE THIS TURN: {turn_objective}"
        if extra_instruction:
            objective_block += f"\n\n{extra_instruction}"
        parts.append(objective_block)

        parts.append(
            "\nRespond in this exact format:\n\n"
            "<thinking>\n[Internal reasoning — 2-4 sentences]\n</thinking>\n\n"
            "<argument>\n[Your debate contribution — 150-300 words]\n</argument>"
        )

        return "\n\n".join(parts)

    def _format_all_turns(self, state: DebateState) -> str:
        lines = [f"FULL DEBATE TRANSCRIPT — Topic: {state.topic}\n"]
        labels = {"us": "US Delegation", "china": "China Delegation", "judge": "EU Judge"}
        for t in state.turns:
            label = labels.get(t.agent, t.agent.upper())
            lines.append(f"\n[{label} — Round {t.round_num}]\n{t.content}")
        return "\n".join(lines)

    def _summarize_round(self, round_num: int, turns: list) -> str:
        turns_text = "\n\n".join(
            f"[{t.agent.upper()}]: {t.content}" for t in turns
        )
        prompt = SUMMARIZE_ROUND.format(round_num=round_num, turns_text=turns_text)
        messages = [{"role": "user", "content": prompt}]
        try:
            return self.tabby_client.chat(messages, temperature=0.3, max_tokens=200)
        except Exception as e:
            print(f"  [summarizer] warning: round {round_num} summary failed: {e}")
            return f"Round {round_num}: {len(turns)} turns completed."

    def _summarize_phase(self, summaries: dict) -> str:
        summaries_text = "\n".join(f"[{k}]: {v}" for k, v in summaries.items())
        prompt = SUMMARIZE_PHASE.format(summaries_text=summaries_text)
        messages = [{"role": "user", "content": prompt}]
        try:
            return self.tabby_client.chat(messages, temperature=0.3, max_tokens=200)
        except Exception as e:
            print(f"  [summarizer] warning: phase summary failed: {e}")
            return f"Phase covering {list(summaries.keys())}."
