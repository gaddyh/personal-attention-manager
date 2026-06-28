# situation_agent.py

import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from personal_attention_manager.agents.situation.core.schemas import SituationInput, SituationOutput, Message
from personal_attention_manager.agents.shared.formatting import format_messages


class SituationAgent:
    def __init__(self, model: str | None = None):
        load_dotenv()

        self.model = model or os.getenv("OPENAI_MODEL")

        llm = ChatOpenAI(
            model=self.model,
            temperature=0,
        )

        self.structured_llm = llm.with_structured_output(
            SituationOutput,
            method="json_schema",
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                You analyze the current situation in a WhatsApp-style chat.

                Your job is not to classify the chat topic.
                Your job is to identify the current actionable state.

                Allowed situation types:
                - question_needs_reply: someone asked a question and expects an answer.
                - user_action_requested: someone requested the user to do an action, such as fix, call, check, review, update, pay, send, book, cancel, or handle something.
                - user_decision_needed: the user needs to choose, approve, confirm, reject, or decide something.
                - user_info_or_artifact_needed: someone is waiting for the user to send information, a file, a document, a link, a photo, a CV, a deck, an invoice, numbers, details, or an answer with factual content.
                - scheduling_coordination: the main issue is setting, moving, confirming, or coordinating a time/date/place.
                - waiting_for_other_person: the user already responded or acted, and the next step is clearly on someone else.
                - fyi_only: information was shared but no response or action is clearly needed.
                - resolved: the open loop appears closed, such as "done", "thanks", "handled", "sent", "fixed", or clear mutual closure.
                - social_or_emotional: social or emotional exchange without a clear concrete action.
                - unclear: not enough context to determine the current situation.

                Core rule:
                Classify the latest open loop, not the general topic.

                Ask:
                1. Who, if anyone, is waiting now?
                2. What are they waiting for?
                3. Is the next expected move on the user or someone else?
                4. Is this still open, or already resolved?

                Important rules:
                - Prefer the latest unresolved request over older context.
                - A short reply like "ok", "done", "sent", or "thanks" may close the loop.
                - If the user says they will do something later, the chat may still be waiting_on_user.
                - If the user asks someone else for something, classify as waiting_for_other_person.
                - If the other person asks the user for a file, document, photo, number, CV, deck, invoice, link, or details, use user_info_or_artifact_needed.
                - If the other person asks the user to choose, approve, confirm, or decide, use user_decision_needed.
                - If the main issue is time/date/place coordination, use scheduling_coordination.
                - If the message is only an FYI and no response is clearly expected, use fyi_only.
                - If the open loop is closed, use resolved even if the earlier message contained a request.
                - If there is no clear actionable state, use unclear.

                Urgency:
                - high: time-sensitive today, blocked work, deadline soon, safety issue, someone explicitly waiting now.
                - medium: needs response/action but not immediate.
                - low: casual, non-urgent, informational, or can wait.

                Output rules:
                - should_user_respond is true only if the user should probably reply or act next.
                - is_waiting_on_user is true when the next expected move is on the user.
                - is_waiting_on_other is true when the next expected move is on someone else.
                - Do not set both is_waiting_on_user and is_waiting_on_other unless the situation is genuinely mutual coordination.
                - confidence above 0.9 only when the open loop is clear.
                - reasoning must be short and user-facing.
                """.strip(),
                ),
                (
                    "human",
                    """
                Analyze this chat.

                Known chat_type, if available:
                {chat_type}

                Recent messages:
                {recent_messages}
                """.strip(),
                ),
            ]
        )

        self.chain = self.prompt | self.structured_llm

    def classify(self, input_data: SituationInput) -> SituationOutput:
        recent_messages_text = format_messages(input_data.recent_messages)

        result: SituationOutput = self.chain.invoke(
            {"recent_messages": recent_messages_text, "chat_type": input_data.chat_type}
        )

        return result


# ----------------------------
# Demo
# ----------------------------

if __name__ == "__main__":
    agent = SituationAgent()

    example = SituationInput(
        recent_messages=[
            Message(
                sender="Dana",
                text="Can you review the doc before I send?",
                sent_time=datetime(2026, 6, 24, 9, 0),
            ),
            Message(
                sender="Me",
                text="Sure, send it over.",
                sent_time=datetime(2026, 6, 24, 9, 1),
            ),
        ]
    )

    output = agent.classify(example)

    print(output)
    print(output.situation_type)
    print(output.confidence)
    print(output.reasoning)
