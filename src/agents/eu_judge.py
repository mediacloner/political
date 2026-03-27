"""EU Judge agent."""

from src.agents.base_agent import BaseAgent
from src.prompts.templates import (
    EU_DEBATER_SYSTEM,
    JUDGE_QUESTION,
    JUDGE_VERDICT,
    build_beliefs_block,
    build_debate_state_block,
    build_recent_history_block,
    build_compressed_past_block,
)


class EUJudge(BaseAgent):
    role = "judge"
    agent_key = "judge"

    def __init__(self, personas: dict, model_cfg: dict, topic: str, persona_name: str = None):
        judge_cfg = personas["judge"]
        persona_name = persona_name or judge_cfg["default_persona"]
        persona = judge_cfg["personas"][persona_name]
        super().__init__(persona, model_cfg, topic)
        self.agent_name = persona_name

    def build_debater_system_prompt(self) -> str:
        """System prompt for EU as active debater (during rounds)."""
        return EU_DEBATER_SYSTEM.format(
            name=self.name,
            title=self.title,
            beliefs_block=build_beliefs_block(self.beliefs),
            debate_style=self.debate_style,
            rhetorical_approach=self.rhetorical_approach,
            core_position=self.core_position,
            topic=self.topic,
        )

    def get_turn_objective(self, round_num: int) -> str:
        """Turn objective for EU as active debater."""
        if round_num == 0:
            return (
                "State the European opening position on this topic. "
                "Identify where both the US and Chinese framings conflict with European "
                "strategic interests. Propose a European alternative framework."
            )
        if round_num % 4 == 0:
            return (
                "European strategic challenge: identify the single argument from either side "
                "that most threatens European interests and directly dismantle it. "
                "Propose a distinctly European solution that neither party has considered."
            )
        return (
            "Advance the European position. Attack the weakest claim made by either the US or China "
            "this round with specific evidence. Identify at least one perspective both sides are "
            "ignoring that matters to European interests."
        )

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
