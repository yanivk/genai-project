# CLAUDE.md

Engineering guide for this repository. Read this before writing any code here.

---

## 1. Project overview

An **SMS-style multi-agent chatbot** that converses with candidates who applied for a
**Python Developer** position. It gathers and verifies information, answers questions about
the role, and drives the conversation to one of two terminal outcomes: an **interview booked**,
or a **polite close**.

The proof of concept runs in **Streamlit** instead of real SMS.

### The three actions

Every bot turn resolves to exactly one action:

| Action | Meaning |
|---|---|
| `continue` | Keep the dialogue going — ask a question, answer a question, exchange information. |
| `schedule` | Actively propose or negotiate interview time slots. |
| `end` | The conversation is finished. **Terminal in both directions**: the interview is confirmed, *or* the candidate opted out. |

### The four agents

| Agent | Responsibility | Backing resource |
|---|---|---|
| **Main Agent** | Orchestrates the turn. Consults the advisors, then decides `continue` / `schedule` / `end` and writes the message sent to the candidate. Owns the conversation memory. | — |
| **Exit Advisor** | Decides whether ending now makes sense, so uninterested candidates are not chased. **Fine-tuned** on the labeled dataset. | OpenAI fine-tuned model |
| **Scheduling Advisor** | Decides whether it is the right moment to schedule; resolves relative dates ("next Friday") against the conversation date; proposes the **3 nearest available slots**. | SQL DB via a LangChain `@tool` |
| **Info Advisor** | Answers questions about the position and keeps the candidate engaged. **Also steers toward the end goal: scheduling an interview.** | Chroma vector DB |

### One turn (from the spec's workflow diagram, `Page 6.png`)

```
candidate message
        │
        ▼
   Main Agent  ── receives and processes input
        │
        ├──────────────┬──────────────┐
        ▼              ▼              ▼
  Exit Advisor   Sched Advisor   Info Advisor
        │              │              │
   end / don't    sched → SQL    info → Vector
                  don't sched    no info needed
        │              │              │
        └──────────────┴──────────────┘
                       ▼
                  Main Agent  ── receives advisor outputs
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
   consult advisor again    send output to candidate
```

The Main Agent may consult the advisors more than once per turn before replying. Guard that
loop with an iteration cap (see `MAX_ADVISOR_ROUNDS` in `app/config.py`) so a turn can never
spin forever.

---

## 2. Golden rules

1. **Never commit secrets.** `.env` is gitignored. `.env.example` holds keys with empty or
   non-secret values only. No API key ever appears in code, notebooks, or prompt files.
2. **`app/config.py` is the single source of truth** for env vars, model ids, and paths.
   No module calls `os.getenv` directly. No module hardcodes a model name or a file path.
3. **Actions are literals.** `"continue"`, `"schedule"`, `"end"` — lowercase, defined once in
   `app/modules/main_agent/actions.py`. Never retype the strings elsewhere.
4. **No business logic in `streamlit_app/`.** The UI collects input, calls `app/`, renders
   output. If you are tempted to put a prompt or a DB query in the UI layer, it belongs in
   `app/modules/`.
5. **Offline steps live in `scripts/` and never run at import time.** Seeding the database,
   building the vector store, and launching a fine-tuning job are explicit manual commands.
   Importing `app` must not hit the network, the OpenAI API, or rebuild an index.
6. **Prompts live in `app/prompts/*.txt`**, never inline in Python.
7. **Follow the course patterns** documented in §4. This project is graded against a course;
   deviating from what was taught needs a reason written down in the code.

---

## 3. Tech stack and pinned versions

Pins come from `../Course24/requirements.txt` — the stack the course actually ran.

```
langchain==0.3.25
langchain-core==0.3.60
langchain-openai==0.3.17
langchain-community==0.3.24
openai==1.81.0
chromadb==1.0.12
streamlit==1.47.1
pydantic==2.11.7
SQLAlchemy==2.0.41
tiktoken==0.9.0
PyPDF2==3.0.1
python-dotenv
pandas / numpy / scikit-learn / seaborn / matplotlib
```

**Why LangChain is pinned to the 0.3.x line.** The Course22 notebooks were authored against
0.3.x but were executed in an environment running LangChain 1.x, where two taught imports are
already broken:

- `from langchain.memory import ChatMessageHistory` → gone in 1.x
- `from langchain.output_parsers.regex import RegexParser` → `ModuleNotFoundError`

Pinning 0.3.x makes every snippet in §4 work verbatim. **Do not bump LangChain** without
re-verifying every import in §4.

Models:

| Purpose | Model |
|---|---|
| Agents and advisors | `gpt-4o-2024-11-20`, `temperature=0` |
| Embeddings | `text-embedding-3-small` (1536 dims) |
| Fine-tuning base | `gpt-4o-mini-2024-07-18` |

---

## 4. Code conventions — the canonical course patterns

Each snippet below is the shape to copy. The source notebook is cited so any deviation can be
checked against the original.

### 4.1 Building an agent — `Course22/LangChain - Agents & Tools.ipynb`

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", system_text),
    MessagesPlaceholder(variable_name="history"),          # only when the agent has memory
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    ("user", "{input}"),
])

llm = ChatOpenAI(model=settings.openai_model, temperature=0)
agent = create_openai_tools_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

reply = executor.invoke({"input": user_input})["output"]   # dict access, NOT .content
```

**The placeholder order is mandatory:** `system` → `history` → `agent_scratchpad` → `user`.
The notebook flags this in bold; reordering breaks tool calling.

Use `create_openai_tools_agent`, **not** `create_tool_calling_agent`, `initialize_agent`,
`bind_tools`, or LangGraph — none of those were taught and none are pinned.

### 4.2 Memory — `Course22/LangChain - Chains & Memory.ipynb`

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store: dict[str, ChatMessageHistory] = {}

def get_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chat_chain = RunnableWithMessageHistory(
    chain,
    get_session_history=get_history,
    input_messages_key="input",
    history_messages_key="history",
)

chat_chain.invoke({"input": text}, config={"configurable": {"session_id": session_id}})
```

Import `ChatMessageHistory` from **`langchain_community.chat_message_histories`** — that is the
import the student notebook had to switch to, and it is stable across both LangChain lines.

Only the **Main Agent** owns memory. Advisors are stateless: they receive the rendered
conversation history as plain text in their `input`.

### 4.3 Tools — `Course22/LangChain - Agents & Tools.ipynb`

```python
from langchain.tools import tool

@tool
def get_available_slots(position: str, from_date: str, limit: int = 3) -> str:
    """Return the nearest available interview slots for a position, starting from a date.

    Args:
        position: Role name as stored in the Schedule table, e.g. "Python Dev".
        from_date: ISO date (YYYY-MM-DD) to search forward from.
        limit: How many slots to return.
    """
```

- The **docstring is the tool description** the model sees. A missing docstring raises
  `ValueError: Function must have a docstring if description not provided.`
- The argument schema is inferred from **type hints** — annotate every parameter.
- Return a **string** the model can read directly. Do not return objects or DataFrames.

### 4.4 Chains — `Course22/LangChain - Models & Parsers.ipynb`

LCEL pipe only:

```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

chain = prompt | llm | JsonOutputParser()
result = chain.invoke({"question": q})     # -> dict
```

Multi-step flows are written as **plain Python functions** that call one chain after another —
never `SequentialChain`, never `LLMChain`.

Gotcha: once `StrOutputParser()` is in the pipe the result is a `str`, so `result.content`
raises `AttributeError`.

### 4.5 Multi-agent orchestration — `Course22`, Example 5

No framework. The Main Agent module exposes a plain Python router that invokes the advisor
executors and dispatches on their JSON verdicts. Keep the routing readable and explicit —
it is the part a grader reads first.

### 4.6 Vector store — `Course23/Embedding & Retrieval.ipynb`, `Course25`

The course used **raw `chromadb` with explicitly computed OpenAI embeddings** — no LangChain
vectorstore wrapper, no `PyPDFLoader`, no `RecursiveCharacterTextSplitter`.

```python
import chromadb
from PyPDF2 import PdfReader

reader = PdfReader(settings.job_description_pdf)
text = "\n".join(p.extract_text() for p in reader.pages if p.extract_text())

response = client.embeddings.create(input=chunks, model=settings.embedding_model)
embeddings = [list(d.embedding) for d in response.data]

chroma = chromadb.PersistentClient(path=str(settings.chroma_path))
collection = chroma.get_or_create_collection(name=settings.chroma_collection)
collection.add(documents=chunks, embeddings=embeddings, ids=ids)
```

Query side:

```python
query_embedding = client.embeddings.create(
    input=[query], model=settings.embedding_model      # SAME model as indexing
).data[0].embedding

results = collection.query(
    query_embeddings=[query_embedding], n_results=3, include=["documents", "distances"]
)
retrieved = results["documents"][0]
```

Two deviations from the course, both deliberate and both required by the spec:

- **`PersistentClient`, not `chromadb.Client()`.** The course used the ephemeral client and
  said so explicitly. The spec requires an *offline* embedding step whose index is reused by
  the app, and Streamlit Cloud needs the index to ship in the repo.
- Chunking is ours to define (the course never chunked). Keep it simple and documented in
  `app/modules/embedding/`.

RAG is hand-wired the way the course taught it: retrieve → stuff the context into the prompt →
call the LLM. No `RetrievalQA`, no `create_retrieval_chain`.

### 4.7 SQL — `Course18/OpenAI API - Basics.ipynb`

```python
import pandas as pd
from sqlalchemy import create_engine, text

engine = create_engine(settings.db_url)
df = pd.read_sql(text("SELECT ... FROM Schedule WHERE ..."), con=engine)
```

Always parameterize — never f-string user input into SQL:

```python
df = pd.read_sql(
    text("SELECT * FROM Schedule WHERE position = :position AND available = 1"),
    con=engine, params={"position": position},
)
```

The course chained `create_engine(...).connect()` immediately, which yields a Connection rather
than an Engine. Do not copy that — keep the Engine and let pandas manage connections.

### 4.8 Environment — Course18–25, verbatim everywhere

```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```

This lives **only** in `app/config.py`. Everywhere else: `from app.config import settings`.

---

## 5. Repository layout

```
final-project/
├── CLAUDE.md                  This file.
├── README.md                  Public project documentation.
├── LICENSE                    MIT.
├── requirements.txt           Pinned dependencies (§3).
├── .env.example               Template for .env — committed, no secrets.
├── .gitignore
├── Page*.png                  The original assignment brief. Reference only.
│
├── app/                       All application code.
│   ├── main.py                CLI entry point — a terminal chat loop.
│   ├── config.py              Env vars, model ids, paths. Single source of truth.
│   ├── prompts/               System prompts as .txt (§8).
│   └── modules/
│       ├── main_agent/        Orchestrator, turn router, action literals.
│       ├── advisors/          exit_advisor.py, scheduling_advisor.py, info_advisor.py.
│       ├── database/          SQLAlchemy engine, Schedule queries, SQLite seeder.
│       ├── embedding/         PDF → chunks → Chroma; retrieval helper.
│       ├── finetuning/        JSONL builder, job launcher, status poller.
│       └── evaluation/        Dataset loader, prediction runner, metrics.
│
├── streamlit_app/             UI only. No business logic.
│   ├── streamlit_main.py      The Streamlit app.
│   └── utils.py               Rendering helpers.
│
├── scripts/                   Offline one-shot commands.
│   ├── seed_database.py       db_Tech.sql → data/tech.db
│   ├── build_vector_store.py  PDF → data/chroma/
│   └── run_finetuning.py      Build JSONL, upload, launch, poll.
│
├── tests/
│   ├── test_main.py           pytest unit tests.
│   └── test_evals.ipynb       Accuracy + confusion matrix (spec deliverable).
│
└── data/
    ├── sms_conversations.json Labeled dataset. Read-only.
    ├── Python Developer Job Description.pdf
    ├── db_Tech.sql            Original T-SQL. Reference only — see §6.2.
    ├── tech.db                Generated SQLite DB. Committed.
    └── chroma/                Persisted Chroma index. Committed.
```

`data/tech.db` and `data/chroma/` are **generated but committed on purpose** — Streamlit
Community Cloud has no way to run the offline steps, so the artifacts must ship.

Every package directory carries an `__init__.py` (required by the spec, `Page 4.png`).

---

## 6. Data contracts

### 6.1 `data/sms_conversations.json`

15 conversations, 103 turns, 59 of them labeled.

```jsonc
[
  {
    "conversation_id": 1,
    "candidate_phone": "+1-555-0201",
    "recruiter_phone": "+1-555-0000",
    "start_time_utc": "2024-04-03T15:12:00Z",
    "turns": [
      { "turn_id": 1, "speaker": "recruiter", "timestamp_utc": "...",
        "text": "...", "label": "continue" },
      { "turn_id": 2, "speaker": "candidate", "timestamp_utc": "...",
        "text": "...", "label": null }
    ]
  }
]
```

- **Only `recruiter` turns are labeled.** Candidate turns always have `label: null`.
- Label distribution: `continue` 25, `schedule` 19, `end` 15. Majority class ≈ **42.4%**.

### Label semantics — read this before writing any prompt

| Label | Fires when | Example |
|---|---|---|
| `continue` | Information is being exchanged — asking about experience, answering a question about the stack or the role. | *"How comfortable are you with SQL in addition to Python?"* |
| `schedule` | A slot is being **proposed or renegotiated**. | *"Our manager can interview you Wednesday at 10 AM or Thursday at 2 PM."* |
| `end` | The conversation is **over**, either way. | Booked: *"Great, your interview is confirmed."* — Declined: *"Understood. I'll close your application for now."* |

Two consequences that are easy to get wrong:

1. **`end` is not a rejection label.** In the dataset it is the last recruiter turn of *every*
   conversation — 11 happy endings and 4 opt-outs. An Exit Advisor trained to equate `end` with
   "candidate is uninterested" will be wrong on most of the class.
2. **Once the candidate accepts a slot, the next turn is `end`, not `schedule`.** Confirming a
   booking closes the conversation.

### Evaluation task definition

Given a conversation history up to and including a candidate turn, predict the label of the
**next recruiter turn**. This is what `tests/test_evals.ipynb` measures.

### 6.2 `Schedule` table

Ported from `data/db_Tech.sql` (T-SQL) to SQLite by `scripts/seed_database.py`. Same schema,
same seeding logic:

| Column | Type | Notes |
|---|---|---|
| `ScheduleID` | INTEGER PK AUTOINCREMENT | |
| `date` | TEXT (`YYYY-MM-DD`) | SQLite has no DATE type; ISO strings sort correctly. |
| `time` | TEXT (`HH:MM:SS`) | |
| `position` | TEXT | `Python Dev`, `Sql Dev`, `Analyst`, `ML` |
| `available` | INTEGER (0/1) | |

Seeding rules, preserved from the original:

- Full year **2024-01-01 → 2024-12-31**
- **Tuesday–Friday and Sunday only** — Saturday and Monday are excluded
- Hourly **09:00–17:00** inclusive (9 slots/day)
- 4 positions; `available` set pseudo-randomly at ~50%

The bot only ever queries `position = 'Python Dev'`.

> **Known dataset/DB mismatch — do not "fix" it.** Several conversations propose **Monday**
> slots (conversations 1, 3, 10), a day the seeded table never contains. The Scheduling Advisor
> must propose only slots that exist in the DB. This affects the wording of generated replies,
> not the evaluated label, so it does not distort the metrics.

### Date resolution

Relative expressions ("next Friday", "Tuesday at 10") resolve against the conversation's
**`start_time_utc`**, not against today's date. Conversations are dated April 2024; using
`datetime.now()` puts every lookup outside the seeded range and returns nothing.

---

## 7. Configuration and secrets

`.env` keys (see `.env.example`):

| Key | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | Required. Never committed. |
| `OPENAI_MODEL` | `gpt-4o-2024-11-20` | Agents and advisors. |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Indexing **and** querying. |
| `FT_BASE_MODEL` | `gpt-4o-mini-2024-07-18` | Fine-tuning base. |
| `FT_EXIT_ADVISOR_MODEL` | *(empty)* | Fine-tuned model id once the job finishes. Empty ⇒ fallback. |
| `DB_URL` | `sqlite:///data/tech.db` | SQLAlchemy URL. |
| `CHROMA_PATH` | `data/chroma` | Persisted index directory. |
| `CHROMA_COLLECTION` | `python_dev_job` | Collection name. |
| `MAX_ADVISOR_ROUNDS` | `2` | Cap on advisor consultations per turn. |

**`config.py` reads `st.secrets` first, then falls back to `os.getenv`.** That single rule is
what lets identical code run locally (`.env`) and on Streamlit Community Cloud (secrets UI).
The `st.secrets` lookup must be wrapped so it degrades quietly when Streamlit is not running.

Paths in `config.py` are absolute, derived from the repo root, so behaviour does not depend on
the current working directory.

---

## 8. Prompt engineering conventions

Structure taught in `Course20/Prompt Engineering.ipynb`. Every file in `app/prompts/` follows
it, in this order:

```
# Identity
Who the agent is and what it decides.

# Instructions
* Numbered or bulleted, imperative.
* State the exact output format.
* State what NOT to do.

# Examples
<conversation id="example-1">
...
</conversation>
<assistant_response id="example-1">
{"action": "schedule", "reason": "..."}
</assistant_response>

# Context
Runtime context goes LAST — it changes per request.
```

- **Context is always the final section**, because it varies per call while the rest is static.
- Few-shot examples use **XML tags carrying `id` attributes**, as in `Course20/few_shot_prompt.txt`.
- Load with `open(path, encoding="utf-8").read()` — never inline a prompt in a `.py` file.
- Every few-shot example is drawn from the **training split** of the dataset (§10), never the
  held-out split.
- **`temperature=0` for every decision-making call.** Only candidate-facing message generation
  may use a higher temperature, and only if it is worth the loss of reproducibility.

---

## 9. Advisor output contract

Advisors return a small JSON object through the taught parser, then get validated:

```python
chain = prompt | llm | JsonOutputParser()      # -> dict
verdict = ExitVerdict.model_validate(chain.invoke({...}))   # pydantic, fail fast
```

`JsonOutputParser` is what the course taught; the Pydantic model is the guard that stops a
malformed LLM response from propagating. `with_structured_output` was not taught and is not used.

| Advisor | Returns |
|---|---|
| **Exit** | `{"should_end": bool, "reason": str}` |
| **Scheduling** | `{"should_schedule": bool, "slots": [{"date": str, "time": str}], "reason": str}` |
| **Info** | `{"info_needed": bool, "answer": str, "sources": [str]}` |

The Main Agent consumes all three verdicts and emits:

```json
{"action": "continue|schedule|end", "message": "text sent to the candidate", "reason": "..."}
```

Define the Pydantic models once in `app/modules/advisors/schemas.py` and import them.

---

## 10. Evaluation rules

Deliverable: `tests/test_evals.ipynb`, following `Course17/ML - Evaluation & Interpretation.ipynb`
and `Course23/Nlp.ipynb`.

1. **Split by `conversation_id`, never by turn.** Turns inside one conversation share a history
   prefix; a turn-level split leaks the test set into training and few-shot examples. With only
   15 conversations, hold out ~5 and make sure both opt-out endings and booked endings appear on
   each side.
2. **Always report the majority-class baseline next to accuracy.** Predicting `continue` for
   everything scores ≈ 42.4%. An accuracy number without that reference point is unreadable.
   The course taught exactly this benchmark idiom.
3. **Confusion matrix, always.** `sklearn.metrics.confusion_matrix` + a `seaborn` heatmap
   (`annot=True, fmt='d', cmap='Blues'`) + a labeled DataFrame with real class names on both axes.
4. **Print the misclassified turns.** Course23 taught explicit false-positive / false-negative
   inspection. Show the conversation id, the history, the true label and the predicted one —
   the error table is worth more than the score.
5. **Report per-class metrics too** (`classification_report`). `end` is the smallest class
   (15 examples); overall accuracy can hide a total failure on it.
6. **Cache LLM predictions to disk.** Re-running the notebook should not re-spend tokens.

---

## 11. Fine-tuning rules

No course precedent — this module is written against the OpenAI SDK directly.

1. **Train on the training split only.** The held-out conversations from §10 never enter the
   JSONL. This is the easiest way to silently invalidate the whole evaluation.
2. Training rows use the chat format, with the same system prompt the Exit Advisor uses at
   inference time:
   ```jsonl
   {"messages":[{"role":"system","content":"..."},{"role":"user","content":"<rendered history>"},{"role":"assistant","content":"{\"should_end\": true, \"reason\": \"...\"}"}]}
   ```
3. **The assistant target is the exact JSON contract from §9** — the fine-tuned model must be a
   drop-in replacement for the prompted one.
4. Both `end` cases must be represented: booked-and-confirmed *and* candidate-opted-out (§6.1).
5. Generated JSONL goes to `data/*.jsonl`, which is **gitignored** — it is derived, and it is
   easy to accidentally commit a version built from the wrong split.
6. The job is launched manually via `scripts/run_finetuning.py`. Write the resulting model id
   into `.env` as `FT_EXIT_ADVISOR_MODEL`.
7. **Always keep the fallback path working.** When `FT_EXIT_ADVISOR_MODEL` is empty, the Exit
   Advisor runs on `OPENAI_MODEL` with few-shot prompting. Never let the app hard-fail because
   a fine-tuning job is missing, expired, or still running.

---

## 12. Common commands

```powershell
# Environment
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Offline steps — run once, in this order
python scripts/seed_database.py          # data/tech.db
python scripts/build_vector_store.py     # data/chroma/
python scripts/run_finetuning.py         # launches the OpenAI job, prints the model id

# Run
python -m app.main                                    # CLI chat loop
streamlit run streamlit_app/streamlit_main.py         # UI

# Test
pytest tests/ -v
jupyter notebook tests/test_evals.ipynb
```

---

## 13. Pitfalls

Ordered by how likely they are to bite.

| # | Pitfall |
|---|---|
| 1 | **`end` is terminal, not negative.** It covers confirmed bookings *and* opt-outs (§6.1). The most common modeling mistake in this project. |
| 2 | **Relative dates resolve against `start_time_utc`**, not `datetime.now()`. The dataset is April 2024; the DB only covers 2024. |
| 3 | **`AgentExecutor.invoke()` returns a dict** — use `response["output"]`, not `.content`. |
| 4 | **Placeholder order in the agent prompt is mandatory**: `system` → `history` → `agent_scratchpad` → `user`. |
| 5 | **`@tool` requires a docstring**, or you get `ValueError: Function must have a docstring if description not provided.` |
| 6 | **Index and query must use the same embedding model.** Mixing `-small` and `-large` yields silently meaningless distances. |
| 7 | **Chroma returns cosine *distance* (0–2), not similarity.** Lower is better. `distance = 1 - cosine_similarity`. |
| 8 | **LangChain 0.3 vs 1.x drift.** `from langchain.memory import ChatMessageHistory` and `langchain.output_parsers` are broken in 1.x — see §3. |
| 9 | **The DB has no Monday or Saturday slots.** The dataset proposes Mondays anyway (§6.2). Propose only what the DB actually has. |
| 10 | **`StrOutputParser` makes the result a `str`** — `.content` on it raises `AttributeError`. |
| 11 | **Do not let `import app` touch the network** or rebuild an index. Offline work belongs in `scripts/`. |
| 12 | **Never commit `.env`, `data/*.jsonl`, or `.streamlit/secrets.toml`.** |
