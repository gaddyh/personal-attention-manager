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
                        - family: relatives, children, parents, spouse, family logistics, emotional family messages
                        - work: job, clients, meetings, projects, tasks, interviews, professional communication
                        - home: house, apartment, bills, rent, mortgage, repairs, maintenance, household logistics
                        - other: anything that does not clearly fit the above

                        Rules:
                        - Use the recent messages as evidence.
                        - Prefer the dominant context, not a single keyword.
                        - If family and home both appear, choose:
                        - family when the main issue is people/relationships/children/parents
                        - home when the main issue is property, bills, repairs, rent, mortgage, maintenance
                        - If confidence is low, still choose the best label and lower the confidence.
                        - Reasoning must be short and user-facing.
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