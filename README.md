# WhatsApp Personal Attention Manager

A backend that listens to a user’s own WhatsApp messages, keeps running state per active chat, and decides which conversations deserve attention first.

This is not just a “chat summarizer.” The core product is an **open-loop and delay-cost detector**:

> Who is waiting for me, why, and what happens if I do not answer soon?

---

## Core Idea

Instead of sending each chat to one big summarizer agent, we split the work into a small cognitive graph.

Each node answers one narrow question:

1. **Who is this?**
2. **What is happening?**
3. **Am I on the hook?**
4. **What happens if I delay?**
5. **How does this user personally care about this?**
6. **Compared to all other chats, what comes first?**

This keeps the system debuggable, controllable, and easier to improve.

---

## System Components

The backend is made of a few separate components. Each component should have a clear responsibility so the agent system does not become tangled with WhatsApp plumbing, login state, or notification policy.

```text
WhatsApp / Green API
   ↓
Webhook endpoints
   ↓
Message storage + running chat history
   ↓
LangGraph agent system
   ↓
ChatCards + priority queue
   ↓
Notification management
```

### 1. Personal WhatsApp Webhook

Endpoint:

```text
POST /personal_green_api/webhook/{user_id}
```

Receives messages from the user’s own WhatsApp connection through Green API.

Responsibilities:

- identify the user by `user_id`
- validate the webhook source
- normalize incoming WhatsApp messages
- append messages to the correct chat history
- mark the chat as dirty / active
- trigger or enqueue background triage work

This endpoint is for the **personal attention manager** use case: listening to the user’s own WhatsApp messages.

### 2. Business Bot Webhook

Endpoint:

```text
POST /bot_business_webhook
```

Receives messages for the business bot flow.

Responsibilities:

- handle customer-facing bot conversations
- route business messages to the bot agent
- keep the business bot logic separate from the personal attention manager

This separation matters. The personal backend listens and prioritizes. The business bot answers customers.

### 3. LangGraph Agent System

The agent system is the cognitive layer. It should not know too much about Green API details.

Responsibilities:

- classify chat identity
- understand the current situation
- detect waiting / obligation
- estimate delay cost
- apply user preferences
- produce `ChatCard`s
- rank waiting chats globally

The mental model is a graph of narrow cognitive steps, not one large summarizer.

### 4. Running Chat History

For each active chat, the system keeps enough state to understand what is currently open.

Stored state may include:

```text
recent raw messages
rolling summary
open loops
last commitments
last inbound timestamp
last outbound timestamp
last ChatCard
known chat type
user corrections
```

The running history is the memory substrate for the agents. The most important part is not the generic summary, but the active **open loops**.

### 5. Green API Login / QR Scan Management

The system needs an operational login layer for connecting each user to Green API.

Responsibilities:

- create or track a Green API instance per user
- expose QR scan login status
- store connection/session state in OLTP storage
- detect whether the user is connected, disconnected, or expired
- avoid running triage for users without a valid WhatsApp connection

This is product infrastructure, not agent logic. Keep it separate.

### 6. Reauthorization Mechanism

Users may need to reconnect or reauthorize their Green API session.

Responsibilities:

- detect expired / disconnected sessions
- mark the user as needing reauth
- notify the user that WhatsApp connection is inactive
- provide a fresh QR scan flow
- pause message processing until reconnection is complete

The agent system should not silently fail because WhatsApp auth expired. Reauth status should be explicit.

### 7. Notification Management

Notification management decides whether to interrupt the user, draft a reply, remind later, or do nothing.

Responsibilities:

- consume ranked `ChatCard`s
- apply deterministic notification policy
- avoid notification spam
- respect snooze / mute / quiet hours
- surface urgent chats
- optionally create draft replies

The LLM explains the situation. The notification layer decides how aggressively to act.

---

## Mental Agent Graph

```text
Raw WhatsApp history
   ↓
1. Identity Agent
   ↓
2. Situation Agent
   ↓
3. Waiting / Obligation Agent
   ↓
4. Delay-Cost Agent
   ↓
5. User-Preference Agent
   ↓
ChatCard
   ↓
6. Global Prioritizer
   ↓
Ordered attention queue
```

Important distinction:

- **Chat-level agents** understand one conversation.
- **Global prioritizer** compares all waiting conversations.

A chat can be important in isolation but still not be the most urgent chat right now.

---

## Agent 1: Identity Agent

Question:

> Who is this person to the user?

Output examples:

```python
chat_type = "family" | "work" | "friend" | "service" | "other" | "unknown"
confidence = 0.0-1.0
should_cache = True | False
```

This is slow-changing. Once confident, cache it.

User correction beats LLM inference.

Example:

```text
User marks chat as "brother"
→ never randomly reclassify it later
```

---

## Agent 2: Situation Agent

Question:

> What is happening right now in this conversation?

This is not priority yet. It creates the semantic scene.

Examples:

```text
pickup logistics
work approval
client scheduling
money/payment
emotional support
FYI
closed conversation
```

Possible output:

```python
situation_type = "pickup_logistics"
current_summary = "They need to know who is picking up the child today."
active_issue = "pickup confirmation"
```

---

## Agent 3: Waiting / Obligation Agent

Question:

> Is someone waiting for me to do something?

This is one of the most important nodes.

A chat may have a topic, but no obligation. The system must understand whether the user is actually expected to respond.

Output:

```python
waiting_on = "me" | "them" | "nobody" | "unclear"
expected_action = "confirm pickup" | "send file" | "answer question" | None
open_loop = True | False
```

Examples:

```text
"Can you confirm by 12?"
→ waiting_on = "me"

"Thanks!"
→ waiting_on = "nobody"

"I’ll send it later"
→ waiting_on = "me" because the user created a commitment
```

---

## Agent 4: Delay-Cost Agent

Question:

> What gets worse if the user does not answer soon?

This is better than asking “is this important?”

Output:

```python
delay_cost = "low" | "medium" | "high" | "critical"
deadline = "today 13:00" | None
reason = "decision blocked" | "person waiting outside" | "emotional escalation" | "no real cost"
```

Examples:

```text
"I’m outside"
→ high / critical

"Can you send this sometime today?"
→ medium

"Look when you have time"
→ low
```

---

## Agent 5: User-Preference Agent

Question:

> Given this user, this person, and this situation, how should it be treated?

The same message can mean different things for different users.

Examples:

```text
Family logistics → interrupt
School direct message → high priority
Work group → low unless directly mentioned
VIP client → boost
Unknown marketing → ignore
```

Output:

```python
preference_weight = "boost" | "neutral" | "deprioritize"
personal_rule = "Family logistics should interrupt"
```

This layer should combine:

- explicit user settings
- user corrections
- observed behavior
- contact-level rules
- learned preferences over time

---

## Agent 6: Global Prioritizer

Question:

> Compared to all waiting chats, what comes first?

Input is not raw WhatsApp history. Input is a list of compact `ChatCard`s.

Example:

```python
[
    ChatCard(chat_id="mom", delay_cost="high", waiting_on="me"),
    ChatCard(chat_id="client", delay_cost="medium", deadline="today"),
    ChatCard(chat_id="work_group", delay_cost="low", waiting_on="nobody"),
]
```

Output:

```text
1. Mom — answer now
2. Client — answer today
3. Work group — no interruption
```

The prioritizer is comparative. It decides attention order, not just per-chat importance.

---

## Central Object: ChatCard

The `ChatCard` is the product-level intelligence object.

Raw messages are too large.
A generic summary is too vague.
The `ChatCard` is compact, rankable, and debuggable.

```python
from pydantic import BaseModel
from typing import Literal


class ChatCard(BaseModel):
    chat_id: str

    # Identity
    chat_type: Literal["family", "work", "friend", "service", "other", "unknown"]
    chat_type_confidence: float

    # Situation
    situation_type: str
    current_summary: str
    active_issue: str | None

    # Obligation
    waiting_on: Literal["me", "them", "nobody", "unclear"]
    expected_action: str | None
    open_loop: bool

    # Urgency
    delay_cost: Literal["low", "medium", "high", "critical"]
    deadline: str | None

    # Personalization
    preference_weight: Literal["boost", "neutral", "deprioritize"]
    personal_rule: str | None

    # Ranking/debugging
    priority_score: int
    suggested_action: Literal[
        "reply_now",
        "draft_reply",
        "remind_later",
        "no_action",
        "needs_user_review",
    ]

    evidence: list[str]
    reason: str
```

---

## Backend Flow

```text
Incoming WhatsApp message
   ↓
Append to raw message history
   ↓
Mark chat as dirty / active
   ↓
Background loop scans active chats round-robin
   ↓
Run chat-level cognitive graph
   ↓
Store updated ChatCard
   ↓
Update priority queue
   ↓
Global prioritizer ranks waiting chats
   ↓
UI / notification / draft layer acts on top items
```

Use round-robin for analysis fairness.
Use a priority queue for attention order.

Those are different queues.

---

## Suggested Storage Model

Per chat, store:

```text
raw_recent_messages
rolling_summary
open_loops
last_commitments
last_inbound_timestamp
last_outbound_timestamp
last_chat_card
known_chat_type
user_corrections
```

The most valuable long-term state is not the summary. It is the set of **open loops**.

Example:

```text
Open loop: Dana is waiting for confirmation about pickup today before 13:00.
```

---

## V1 Implementation

Start with fewer physical agents, but keep the mental separation clear.

Recommended V1:

```text
1. Identity Agent
2. Situation + Waiting Agent
3. Delay-Cost Agent
4. Preference / Scoring Agent
5. Global Prioritizer
```

Recommended V2:

```text
Split Situation and Waiting into separate agents
```

Why?

The likely failure mode is:

```text
The system understands the topic, but fails to know whether the user is actually expected to answer.
```

If that happens often, Waiting / Obligation deserves its own node.

---

## Priority Philosophy

Do not ask only:

```text
Is this important?
```

Ask:

```text
Is someone waiting for me?
What action is expected?
What happens if I delay?
Is there a deadline?
How does this user usually treat this type of situation?
Compared to other waiting chats, what comes first?
```

This makes the system smarter and easier to evaluate.

---

## MVP Priority Levels

Use simple levels first:

```text
P0 — critical now
P1 — answer soon
P2 — answer today
P3 — low priority
P4 — no action
```

Examples:

```text
P0:
- "I’m outside"
- "Are you picking up the child?"
- "Call me now"

P1:
- "Can you approve this before noon?"
- "Client is waiting"

P2:
- "Can you send this today?"
- "Please confirm when possible"

P3:
- "FYI"
- "Look when you have time"

P4:
- closed conversations
- spam
- reactions
- generic group noise
```

---

## Design Principle

The LLM should explain the situation.
The backend should control the queue behavior.

That means:

- LLM produces structured understanding.
- Backend applies deterministic policy.
- User feedback improves future preferences.
- Every priority decision remains inspectable.

---

## Final Framing

This system is a personal attention manager for WhatsApp.

Its job is not to summarize everything.
Its job is to decide:

> Who needs me, why, how soon, and what should I do first?
