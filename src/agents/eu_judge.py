"""EU Judge agent."""

from src.agents.base_agent import BaseAgent
from src.prompts.templates import (
    JUDGE_QUESTION,
    JUDGE_VERDICT,
    build_debate_state_block,
    build_recent_history_block,
    build_compressed_past_block,
)


class EUJudge(BaseAgent):
    role = "judge"
    agent_key = "judge"

    def __init__(self, personas: dict, model_cfg: dict, topic: str):
        persona = personas["judge"]
        super().__init__(persona, model_cfg, topic)

    def build_question_messages(self, state) -> list[dict]:
        """Build messages for the inter-round question turn."""
        system = self.build_system_prompt()
        tier2 = build_debate_state_block(state)
        tier3 = build_recent_history_block(state.get_recent_turns(4))
        tier4 = build_compressed_past_block(state.summaries)

        parts = []
        if tier2:
            parts.append(tier2)
        if tier4:
            parts.append(tier4)
        if tier3:
            parts.append(tier3)
        parts.append(
            "---\nYOUR TASK: Generate a targeted question for the debater whose last argument "
            "contained the weakest logical support or most unexamined assumption.\n\n"
            "Respond in this format:\n\n"
            "<thinking>\n[Which argument was weakest and why?]\n</thinking>\n\n"
            "<question>\n[Your targeted question — 50-100 words. Direct and specific.]\n</question>"
        )

        user_content = "\n\n".join(parts)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    def build_verdict_messages(self, state) -> list[dict]:
        """Build messages for the final verdict."""
        system = self.build_system_prompt()

        # For verdict, pass the full transcript
        turns_text = ""
        labels = {"us": "US Delegation", "china": "China Delegation", "judge": "EU Judge"}
        for t in state.turns:
            label = labels.get(t.agent, t.agent.upper())
            turns_text += f"\n\n[{label} — Round {t.round_num}]\n{t.content}"

        debate_state_block = build_debate_state_block(state)

        user_content = (
            f"FULL DEBATE TRANSCRIPT\nTopic: {state.topic}\n"
            f"{debate_state_block}\n\n"
            f"{turns_text}\n\n"
            "---\n"
            "Deliver your final assessment following the 4-step structured analysis:\n"
            "STEP 1: Steelman both sides\n"
            "STEP 2: Score each participant (Evidence Quality, Logical Coherence, Persuasiveness, Engagement — 1-5)\n"
            "STEP 3: Identify blind spots (perspectives neither side raised)\n"
            "STEP 4: Deliver your verdict framed around European strategic interests\n\n"
            "Do NOT manufacture false balance. If one side is clearly stronger, say so.\n\n"
            "Respond in this format:\n\n"
            "<thinking>\n[What tipped the scales?]\n</thinking>\n\n"
            "<verdict>\n[Complete 4-step assessment — 400-600 words]\n</verdict>"
        )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    def judge_says_done(self, verdict_text: str) -> bool:
        """
        Check if the judge's verdict signals the debate is truly resolved.
        Simple heuristic: look for conclusive language.
        """
        conclusive_phrases = [
            "clear winner", "clearly superior", "definitively", "unambiguously",
            "the evidence strongly favours", "the evidence strongly favors",
            "i recommend", "europe should", "the verdict is",
        ]
        lower = verdict_text.lower()
        return any(phrase in lower for phrase in conclusive_phrases)
