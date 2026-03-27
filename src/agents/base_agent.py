"""
BaseAgent — shared logic for all three debate agents.
"""

import json
from src.prompts.templates import (
    DEBATER_SYSTEM,
    JUDGE_SYSTEM,
    build_beliefs_block,
    extract_tag,
    CLAIM_EXTRACTION,
)


class BaseAgent:
    role = "debater"  # Override in subclass: "debater" | "judge"

    def __init__(self, persona: dict, model_cfg: dict, topic: str):
        self.name = persona["name"]
        self.title = persona["title"]
        self.beliefs = persona["beliefs"]
        self.debate_style = persona["debate_style"]
        self.rhetorical_approach = persona["rhetorical_approach"]
        self.core_position = persona["core_position"]
        self.topic = topic
        self.model_cfg = model_cfg

    def build_system_prompt(self) -> str:
        template = JUDGE_SYSTEM if self.role == "judge" else DEBATER_SYSTEM
        return template.format(
            name=self.name,
            title=self.title,
            beliefs_block=build_beliefs_block(self.beliefs),
            debate_style=self.debate_style,
            rhetorical_approach=self.rhetorical_approach,
            core_position=self.core_position,
            topic=self.topic,
        )

    def parse_response(self, raw: str) -> tuple[str, str]:
        """
        Returns (argument, thinking).
        Falls back gracefully if tags are missing.
        """
        argument = extract_tag(raw, "argument")
        thinking = extract_tag(raw, "thinking")

        # If no <argument> tag found, use full response as argument
        if argument == raw.strip():
            thinking = ""

        return argument, thinking

    def parse_question(self, raw: str) -> tuple[str, str]:
        """For judge question turns: returns (question, thinking)."""
        question = extract_tag(raw, "question")
        thinking = extract_tag(raw, "thinking")
        if question == raw.strip():
            thinking = ""
        return question, thinking

    def parse_verdict(self, raw: str) -> tuple[str, str]:
        """For judge verdict: returns (verdict, thinking)."""
        verdict = extract_tag(raw, "verdict")
        thinking = extract_tag(raw, "thinking")
        if verdict == raw.strip():
            thinking = ""
        return verdict, thinking

    def extract_claims(self, argument: str, tabby_client) -> tuple[list, list]:
        """
        Use the LLM to extract structured claims and evidence citations.
        Returns (claims, evidence). Fails gracefully.
        """
        prompt = CLAIM_EXTRACTION.format(argument_text=argument)
        messages = [{"role": "user", "content": prompt}]
        try:
            raw = tabby_client.chat(messages, temperature=0.2, max_tokens=300)
            # Find JSON block
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(raw[start:end])
                return data.get("claims", []), data.get("evidence", [])
        except Exception:
            pass
        return [], []
