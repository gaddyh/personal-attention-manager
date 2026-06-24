# chat_type_agent.py

import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from personal_attention_manager.agents.classification.core.schemas import ClassificationInput, ClassificationOutput, Message


class ChatTypeAgent:
    def __init__(self, model: str | None = None):
        load_dotenv()

        self.model = model or os.getenv("OPENAI_MODEL")

        llm = ChatOpenAI(
            model=self.model,
            temperature=0,
        )

        self.structured_llm = llm.with_structured_output(
            ClassificationOutput,
            method="json_schema",
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
            You classify a WhatsApp-style chat into exactly one type.

            Allowed chat types:
            - family: relatives, children, parents, spouse, grandparents, family care, child logistics, emotional family coordination
            - work: job, clients, colleagues, meetings, projects, tasks, interviews, incidents, deliverables, invoices, approvals, professional coordination
            - home: house, apartment, bills, rent, mortgage, repairs, maintenance, utilities, landlord, building committee, property logistics
            - other: casual/social, marketplace, buying/selling/giveaway, travel, hobbies, restaurants, entertainment, personal errands, personal tech issues, or anything that does not clearly fit family/work/home

            Core rule:
            Classify by dominant intent, not by surface keywords.

            Ask yourself:
            "What is the actual thing being coordinated or discussed?"

            Important negative rules:
            - Household objects do not automatically mean home.
            If the chat is about buying, selling, giving away, pickup, availability, fitting an item in a car/elevator, or "still available", classify as other unless there is actual household maintenance, bill, rent, utility, repair, or property coordination.

            - Technical or work-like words do not automatically mean work.
            Choose work only when the dominant purpose is professional coordination: a task, deliverable, client, colleague workflow, incident, approval, deadline, invoice, hiring process, or business outcome.
            If technical/work words are used jokingly or for a personal device/hobby, classify as other.

            - Family members or family-like objects do not automatically mean family.
            Choose family only when the dominant purpose is family care, child logistics, parent/grandparent/sibling coordination, emotional support, or family relationship management.
            If a relative is merely mentioned in a marketplace, social, moving, or casual context, classify as other.

            Tie-breaking:
            - If family and home both appear:
            - choose family when the dominant issue is people, relationships, children, parents, grandparents, or care
            - choose home when the dominant issue is property, bills, repairs, rent, utilities, apartment, or building logistics

            - If work and casual/social both appear:
            - choose work only when the dominant purpose is professional
            - choose other when work is only background, slang, a joke, or casual context

            - If home-like objects appear in a marketplace/sale/pickup context:
            - choose other

            - If family-like objects appear in a marketplace/sale/pickup context:
            - choose other

            Work-critical rule:
            If the chat contains a concrete professional action request, classify as work even if the conversation starts socially.

            Concrete professional actions include:
            - review a CV/resume before sending
            - review candidate notes
            - review a contract, invoice, deck, report, proposal, bug notes, or numbers
            - approve, send, update, fix, check, sanity-check, or provide feedback on a work/career artifact
            - coordinate with legal, finance, client, hiring, vendor, QA, ops, or another professional process

            A social wrapper such as coffee, drinks, lunch, "hey", "random q", or joking tone does not override a concrete professional request.

            Confidence calibration:
            - Use confidence > 0.9 only when there is no serious competing label.
            - If the chat contains misleading surface cues, use lower confidence, usually 0.65–0.85.
            - If the label depends on subtle intent, do not output very high confidence.
            - If confidence is low, still choose the best label and lower the confidence.

            Reasoning:
            - Keep reasoning short and user-facing.
            - Mention the dominant intent.
            - If there is a misleading cue, explain why it did not determine the label.
            """.strip(),
                    ),
                    (
                        "human",
                        """
            Classify this chat:

            {recent_messages}
            """.strip(),
                    ),
                ]
            )

        self.chain = self.prompt | self.structured_llm

    def classify(self, input_data: ClassificationInput) -> ClassificationOutput:
        recent_messages_text = self._format_messages(input_data.recent_messages)

        result: ClassificationOutput = self.chain.invoke(
            {"recent_messages": recent_messages_text}
        )

        return result

    @staticmethod
    def _format_messages(messages: list[Message]) -> str:
        if not messages:
            return "No recent messages."

        sorted_messages = sorted(messages, key=lambda m: m.sent_time)

        lines = []
        for msg in sorted_messages:
            lines.append(
                f"[{msg.sent_time.isoformat()}] {msg.sender}: {msg.text}"
            )

        return "\n".join(lines)


# ----------------------------
# Demo
# ----------------------------

if __name__ == "__main__":
    agent = ChatTypeAgent()

    example = ClassificationInput(
        recent_messages=[
            Message(
                sender="Dana",
                text="Can we move the client meeting to tomorrow?",
                sent_time=datetime(2026, 6, 24, 9, 0),
            ),
            Message(
                sender="Me",
                text="Yes, but I need to finish the production bug first.",
                sent_time=datetime(2026, 6, 24, 9, 3),
            ),
        ]
    )

    output = agent.classify(example)

    print(output)
    print(output.chat_type)
    print(output.chat_type_confidence)
    print(output.reasoning)