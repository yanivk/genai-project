# Engineering guide

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
| **Scheduling Advisor** | Decides whether it is the right moment to schedule; asks the candidate **when they are free**, resolves their relative dates ("next Friday") against the conversation date, and returns the slots the schedule actually holds for that window (§4.12). | SQL DB via a LangChain `@tool` |
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

**Python 3.13, and it is not a preference.** None of these pins publish wheels for 3.14, so
pip falls back to building from source: `tiktoken` needs a Rust toolchain, `cffi` needs
`ffi.h`, and neither exists on a Streamlit Cloud builder. That is exactly how the first
deploy failed. On Community Cloud the version is chosen under *Advanced settings* at
creation time and **cannot be changed afterwards** — a wrong choice means deleting the app
and redeploying.

**Dependencies are split, and the split is load-bearing.** `requirements.txt` holds only
what the app imports at runtime, because that is the file Streamlit Cloud installs;
`requirements-dev.txt` adds the notebook, the tests and `PyPDF2` for the offline scripts.
A test in `tests/test_main.py` imports the whole runtime path with every dev-only package
blocked, so a stray top-level `import matplotlib` under `app/` fails the suite instead of
the deploy.

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

No framework. `app/modules/main_agent/orchestrator.py` exposes a plain Python router that
invokes the advisors and dispatches on their JSON verdicts.

**The action is chosen in code, not by the LLM.** `resolve_action()` applies a fixed
precedence — **exit beats schedule beats continue** — and the Main Agent is then *told* that
action and writes the candidate-facing message for it. Two reasons:

1. Asked to weigh the verdicts itself, the model reliably picked `schedule` over `end`
   whenever both advisors fired — which is exactly the case where the candidate has just
   accepted a time, so the answer is always `end`. Moving the choice into code took `end`
   recall from 0.40 to 1.00.
2. It matches the course's own multi-agent example, where dispatch is a plain Python
   function rather than a model call.

Deciding first and writing second also means the wording can never contradict the action.

### 4.9 Prompt files carry literal JSON — never `str.format` them

The prompt files contain JSON braces in their few-shot examples, so `str.format` and
`ChatPromptTemplate` would both try to read `{"should_end": true}` as a template field.

Two rules, both implemented in `app/modules/common.py`:

- Substitute placeholders with `fill()`, which uses `str.replace`.
- Pass the rendered text into the chain as the **value** of a single `{system_text}`
  variable. LangChain substitutes variables without re-parsing what it substituted, so the
  braces inside survive:

```python
prompt = ChatPromptTemplate.from_messages([("system", "{system_text}"), ("user", "{input}")])
chain = prompt | llm | JsonOutputParser()
chain.invoke({"system_text": fill(load_prompt("exit_advisor"), conversation=text),
              "input": "Return your JSON verdict for the conversation above."})
```

### 4.10 Shared helpers — `app/modules/common.py`

`load_prompt`, `fill`, `get_llm`, `get_openai_client`, `render_history`, `parse_json_output`.
Written once so the patterns above are not copy-pasted into four advisor modules. Import
from here rather than re-deriving.

### 4.11 Advisors degrade, they do not raise

Every advisor wraps its model call and returns a **conservative default** on failure —
`should_end=False`, `should_schedule=False`, `info_needed=False` — logging the exception.
One flaky call must not end a candidate's conversation or break a turn.

The cost of that safety is that a real bug can hide as a quiet degradation. Two were found
this way during the first evaluation run: a Chroma client race under thread fan-out, and an
empty-string embedding request on a conversation's opening turn. **When retrieval or an
advisor looks inert, check the logs before concluding the model is at fault.**

### 4.12 Scheduling is availability-first — the candidate names the window

The bot never opens with dates of its own. Once the screening gate is passed it asks *when
the candidate is free*, and only the turn after does it read the schedule — for the window
they gave. Two turns, both `schedule`:

| Turn | Verdict | Message |
|---|---|---|
| 1 | `should_schedule=true`, `needs_availability=true`, `slots=[]` | *"Take a look at the availability calendar next to this chat and tell me which day suits you."* |
| 2 | `should_schedule=true`, `needs_availability=false`, `slots=[…]` | *"Tuesday afternoon works — the 1st at 12, 2 or 3 PM."* |

That first message points the candidate at the sidebar calendar in `streamlit_app/`, which
is drawn from the same `Schedule` table the advisor queries — so what they are asked to pick
from is what the bot can actually book.

The candidate's own words become the tool's arguments: `when` / `until` (pass the same value
twice for one single day) and `time_of_day` (`morning`, `midday`, `afternoon`, `evening`,
carved out of 09:00–17:00 because nothing else is seeded). An empty result is a normal
answer — the candidate may have named a Monday — and the advisor then widens the window
rather than inventing something.

Three code-level guards, because none of this can be left to the prompt alone:

- **`reconcile()`** keeps `needs_availability` and `slots` mutually exclusive. A
  `should_schedule` turn with neither is an *ask*: left alone, the Main Agent has a
  scheduling turn with nothing to propose, and it fills that gap by inventing a time.
- **`validate_slots()`** re-reads every proposed slot from the database and drops what is
  not free, substituting the schedule's real openings from the same day. This is not
  hypothetical: asked for a Monday, the advisor calls the tool correctly, sees
  `12:00 / 14:00 / 15:00`, and still reports `09:00 / 10:00 / 11:00`. Tightening the prompt
  did not stop it. The candidate would be offered three interviews the calendar cannot
  honour, and every layer above would believe them.
- The **Exit Advisor** must not read a stated availability as a booking. *"Thursday at 2 PM
  would be perfect"*, said in answer to *"when are you free?"*, is the input to a lookup —
  nothing has been checked yet. A counter-proposal made *after* slots were offered still
  ends the conversation, which is what the dataset means by `end` (§6.1).

That last rule costs one labeled turn: conversation 3, where the candidate names a time
before anything was offered and the next recruiter turn is `end`. It sits in the **training**
split, so the held-out score is untouched — but it is a real §10.7 trade-off, not a free one.

**Known limitation.** A counter-proposal after slots were offered is confirmed without a
database check, because the dataset labels that turn `end` and the Exit Advisor has no tool.
Fixing it properly costs three held-out `end` turns; it was left alone deliberately.

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

**Chroma is the vector database, in every code path.** It is a course requirement, not an
implementation detail to optimise away. Indexing goes through `collection.add`, retrieval
through `collection.query`.

Two deviations from the course, both deliberate:

- **`PersistentClient`, not `chromadb.Client()`.** The course used the ephemeral client and
  said so explicitly. The spec requires an *offline* embedding step whose index is reused by
  the app.
- Chunking is ours to define (the course never chunked). Keep it simple and documented in
  `app/modules/embedding/`.

**The offline step writes two artifacts** (`indexer.build_index`):

| Artifact | Size | Committed? |
|---|---|---|
| `data/chroma/` — the persisted Chroma database | ~60 MB | **No**, gitignored |
| `data/vector_store.json` — same chunks + vectors as JSON | ~94 KB | **Yes** |

Chroma pre-allocates its HNSW index for 10,000 vectors, so it writes 60 MB for our three
chunks — too heavy for a repo. `retriever.get_collection()` therefore resolves in two steps:
use `data/chroma/` when it exists, otherwise **rebuild that Chroma database from the JSON
seed** and use it. A fresh checkout and Streamlit Cloud both take the second path.

The seed is a transport format, never a query path. Nothing reads vectors out of the JSON.

Resetting the persisted store uses Chroma's `delete_collection`, **not** `shutil.rmtree`: on
Windows the store keeps file handles open, so `rmtree` raises `PermissionError` and leaves a
half-removed directory behind.

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
├── ENGINEERING.md             This file.
├── README.md                  Public project documentation.
├── LICENSE                    Apache-2.0.
├── .python-version            3.13. The pins have no 3.14 wheels (§3).
├── requirements.txt           Runtime dependencies only — what Cloud installs (§3).
├── requirements-dev.txt       The above, plus notebook, tests and offline scripts.
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
    ├── vector_store.json      Chunks + vectors seed. Committed (§4.6).
    ├── chroma/                Persisted Chroma DB. Generated, gitignored.
    └── cache/                 Cached eval predictions. Gitignored.
```

`data/tech.db` and `data/vector_store.json` are **generated but committed on purpose** —
Streamlit Community Cloud has no way to run the offline steps, so those artifacts must ship.
`data/chroma/` is rebuilt from the seed on first use instead (§4.6).

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
    "start_time_utc": "2026-07-15T15:12:00Z",
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

- **Tuesday–Friday and Sunday only** — Saturday and Monday are excluded
- Hourly **09:00–17:00** inclusive (9 slots/day)
- 4 positions; `available` set pseudo-randomly at ~50%
- Range **2026-01-01 → 2027-12-31**

The bot only ever queries `position = 'Python Dev'`.

The date range is the one rule that does not come from the original script. It must always
cover every conversation in the dataset plus a margin of future slots — see §6.3.

### 6.3 Dates: the dataset and the schedule move together

Both were originally set in **2024**. They were shifted to the present so the system can be
tested against live dates, by a **whole number of weeks (119)** — which preserves every
weekday, and therefore the meaning of every relative expression in the conversation text
("this Friday", "next Tuesday").

Two invariants follow. Break either one and the Scheduling Advisor silently finds nothing:

1. **The seeded range must cover the dataset**, with future slots left over.
   Dataset spans **2026-07-15 → 2026-08-11**; the schedule spans **2026-01-01 → 2027-12-31**.
2. **No message may propose a Monday or a Saturday.** The schedule has no such rows. The
   dataset originally proposed Mondays in 10 turns; those were rewritten to Tuesday, or to
   Wednesday where the same message already mentioned Tuesday. Labels were untouched.

Both invariants are enforced by tests in `tests/test_main.py`, so a future re-shift that
breaks one fails the suite rather than degrading silently. If you do shift again: shift by
whole weeks, re-seed the database over the new range, and re-run the tests.

**Resolution rule.** Relative expressions ("next Friday", "Tuesday at 10") resolve against
the conversation's **`start_time_utc`**, not against today's date. Using `datetime.now()`
for a historical conversation puts every lookup in the wrong week and returns the wrong
slots. Live chats pass today as the anchor, which is why the schedule extends into 2027.

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
| `CHROMA_PATH` | `data/chroma` | Persisted Chroma DB directory (generated, gitignored). |
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
- `# Examples` is the one optional section. `exit_advisor_finetuned.txt` omits it on purpose
  (§11.8) — the fine-tuned model learned the behaviour those examples were teaching.
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
| **Scheduling** | `{"should_schedule": bool, "needs_availability": bool, "slots": [{"date": str, "time": str}], "reason": str}` |
| **Info** | `{"info_needed": bool, "answer": str, "sources": [str]}` |

`needs_availability` is the availability-first flow of §4.12: `should_schedule` with an empty
`slots` list means *ask them when they are free*, not *propose nothing*.

`resolve_action()` turns the first two into the action (§4.5). The Main Agent then emits:

```json
{"action": "continue|schedule|end", "message": "text sent to the candidate", "reason": "..."}
```

The `action` it returns is **overwritten** with the resolved one — the model only writes the
message. Define the Pydantic models once in `app/modules/advisors/schemas.py` and import them.

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
   The cache is keyed by decision point, so it also stores `prediction_fingerprint()` — a
   hash of the four prompts and the two model ids. Change a prompt and the cache is
   discarded instead of replayed. Without it, tuning a prompt and re-running reports the
   *previous* score as if it were the new one, and nothing in the output says so.
7. **The metric is not the product.** The dataset's recruiters propose an interview after
   one answer about the candidate's background, so a bot that does the same scores well —
   and books the hiring manager's time with someone whose Python experience was never
   established. The Scheduling Advisor therefore holds a screening gate (§13, pitfall 16)
   that costs about one test turn. When a prompt change trades accuracy for behaviour,
   say so next to the number rather than quietly taking the higher score.

---

## 11. Fine-tuning rules

No course precedent — the spec requires it anyway (`page 3.png`: *"Conversation Exit Advisor
should be fine-tuned"*, `Page 5.png`: *"OpenAI account — API keys for embeddings, chat, and
fine-tuning"*). It is an OpenAI job, not a scikit-learn `fit`, and this module is written
against the OpenAI SDK directly.

1. **Train on the training split only.** The held-out conversations from §10 never enter the
   JSONL. This is the easiest way to silently invalidate the whole evaluation.
2. **A training row is an inference call.** The conversation lives in the *system* prompt's
   `# Context` section (§8), so the row is `system` = the rendered prompt, `user` = the same
   fixed directive the advisor sends, `assistant` = the target JSON:
   ```jsonl
   {"messages":[{"role":"system","content":"# Identity ... # Context\n<rendered history>"},{"role":"user","content":"Return your JSON verdict for the conversation above."},{"role":"assistant","content":"{\"should_end\":true,\"reason\":\"...\"}"}]}
   ```
   Both halves come from `exit_advisor.build_system_text()` and `exit_advisor.DIRECTIVE`,
   called rather than re-typed — train/serve skew here would be invisible and expensive.
3. **The assistant target is the exact JSON contract from §9** — the fine-tuned model must be a
   drop-in replacement for the prompted one.
   `reason` has no ground truth in the dataset, so it is generated deterministically from the
   situation (`dataset._reason_for`). It is explanatory only — `resolve_action()` reads the
   boolean — but the model is trained to emit it, so it must be consistent.
4. Both `end` cases must be represented: booked-and-confirmed *and* candidate-opted-out (§6.1).
5. Generated JSONL goes to `data/*.jsonl`, which is **gitignored** — it is derived, and it is
   easy to accidentally commit a version built from the wrong split.
6. The job is launched manually via `scripts/run_finetuning.py`. Write the resulting model id
   into `.env` as `FT_EXIT_ADVISOR_MODEL`.
7. **Always keep the fallback path working.** When `FT_EXIT_ADVISOR_MODEL` is empty, the Exit
   Advisor runs on `OPENAI_MODEL` with few-shot prompting. Never let the app hard-fail because
   a fine-tuning job is missing, expired, or still running.
8. **The prompt follows the model — two files, one contract.**

   | `FT_EXIT_ADVISOR_MODEL` | Model | Prompt file |
   |---|---|---|
   | empty | `OPENAI_MODEL` | `exit_advisor.txt` — 7 worked examples |
   | set | the fine-tuned id | `exit_advisor_finetuned.txt` — identity, instructions, no examples |

   Replacing few-shot examples with learned behaviour is the *reason* to fine-tune; carrying
   them into every training row would pay for them twice and teach nothing. `build_system_text()`
   picks the file from `settings.is_finetuned`, and the dataset builder forces `finetuned=True`,
   so training and serving cannot drift apart. The output contract is identical in both files —
   that is what makes the swap a drop-in.
9. **The job cannot actually be launched — OpenAI closed the platform.** Attempting it returns:

   ```
   403 training_not_available — "OpenAI is winding down the fine-tuning platform and your
   organization is no longer able to create new fine-tuning training jobs."
   ```

   Announced 2026-05-07: organizations that had never run a job before that date are blocked
   immediately; everyone else loses access on 2027-01-06. Inference on already-trained models
   survives until the base model retires. This organization never ran one, so `create_job()` is
   unreachable here — the assignment predates the wind-down.

   Consequences for anyone working on this repo:

   - **Do not debug it.** The 403 is not caused by the JSONL, the split, the base model or the
     key. `create_job()` translates it into `FineTuningUnavailable` precisely so it is not
     mistaken for a data bug.
   - Everything up to job creation is real and verified: the split, the JSONL, and the upload
     (`upload_training_file()` returns a file id). `--dry-run` exercises the whole builder
     offline.
   - **The few-shot fallback is now the only path**, which is what rule 7 was written for. It is
     what the reported evaluation measures, and `end` already scores 1.00/1.00 on the held-out
     split — so the fine-tuned model was never where the remaining headroom was.
   - `FT_EXIT_ADVISOR_MODEL` stays empty. Keep both prompt files and the model swap working: if
     a fine-tuned id ever becomes available, it is a one-line `.env` change.

---

## 12. Common commands

```powershell
# Environment — 3.13, see section 3
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt      # runtime deps + notebook + tests
git config core.hooksPath .githooks       # once per clone — see below

# Offline steps — run once, in this order
python scripts/seed_database.py          # data/tech.db
python scripts/build_vector_store.py     # data/chroma/
python scripts/run_finetuning.py --dry-run   # build + inspect the JSONL, spend nothing
python scripts/run_finetuning.py             # launches the OpenAI job, prints the model id

# Run
python -m app.main                                    # CLI chat loop
streamlit run streamlit_app/streamlit_main.py         # UI

# Test
pytest tests/ -v
jupyter notebook tests/test_evals.ipynb
```

### Commit co-authorship

`git config core.hooksPath .githooks` enables `prepare-commit-msg`, which appends a
`Co-Authored-By` trailer for each of the other two team members. The work is done by
three people but pushed from one machine; without the trailers GitHub credits the entire
history to one account.

The addresses are the GitHub `noreply` form (`<id>+<login>@users.noreply.github.com`).
Neither account publishes a real address, and this is the form GitHub itself resolves
back to a profile. The hook skips the current committer, so nobody is listed as their
own co-author, and it is idempotent — `git commit --amend` does not stack duplicates.

The hook also strips any `Co-Authored-By: Claude` trailer. Assistant tooling adds one
automatically, and it surfaces in the repository's contributor list; the credit belongs to
the three people on the team.

**The hook is opt-in per clone.** Git refuses to run hooks from a checked-in directory
unless `core.hooksPath` says so, which is a security feature, not an obstacle to route
around. A missing trailer means someone skipped that config line.

---

## 13. Pitfalls

Ordered by how likely they are to bite.

| # | Pitfall |
|---|---|
| 1 | **`end` is terminal, not negative.** It covers confirmed bookings *and* opt-outs (§6.1). The most common modeling mistake in this project. |
| 2 | **Relative dates resolve against `start_time_utc`**, not `datetime.now()`. The dataset spans 2026-07-15 → 2026-08-11; the schedule covers 2026-2027 (§6.3). |
| 3 | **`AgentExecutor.invoke()` returns a dict** — use `response["output"]`, not `.content`. |
| 4 | **Placeholder order in the agent prompt is mandatory**: `system` → `history` → `agent_scratchpad` → `user`. |
| 5 | **`@tool` requires a docstring**, or you get `ValueError: Function must have a docstring if description not provided.` |
| 6 | **Index and query must use the same embedding model.** Mixing `-small` and `-large` yields silently meaningless distances. |
| 7 | **Chroma returns cosine *distance* (0–2), not similarity.** Lower is better. `distance = 1 - cosine_similarity`. |
| 8 | **LangChain 0.3 vs 1.x drift.** `from langchain.memory import ChatMessageHistory` and `langchain.output_parsers` are broken in 1.x — see §3. |
| 9 | **The DB has no Monday or Saturday slots**, and none exist outside 09:00–17:00. Propose only what the DB actually returns; never invent a plausible-sounding time (§6.3). |
| 10 | **`StrOutputParser` makes the result a `str`** — `.content` on it raises `AttributeError`. |
| 11 | **Do not let `import app` touch the network** or rebuild an index. Offline work belongs in `scripts/`. |
| 12 | **Never commit `.env`, `data/*.jsonl`, or `.streamlit/secrets.toml`.** |
| 13 | **`chromadb.PersistentClient` is not thread-safe to construct.** Concurrent builds race on the Rust bindings and fail with a misleading "tenant default_tenant does not exist". `get_collection()` serialises the first construction behind a lock — go through it, never build a client inline. |
| 14 | **The embeddings endpoint rejects an empty string** with a 400. A conversation's opening turn has no candidate message, so guard the query before embedding it. |
| 15 | **Advisors swallow their own errors by design** (§4.11), so a broken dependency looks like a lazy model. Read the logs before tuning a prompt. |
| 16 | **The Scheduling Advisor holds a screening gate: no interview until a PYTHON-SPECIFIC answer exists.** A job title is not one — *"fullstack developer for 10 years"* says nothing about Python, and the role needs 3+ years of it. Loosening this raises the eval score and produces a bot that books interviews it should not. |
| 17 | **The eval cache is fingerprinted (§10.6).** If you bypass `predict_all` to save time, you lose that guard and a stale run will report the old score under the new prompts. |
| 18 | **A `schedule` turn with an empty `slots` list is a question, not a bug** — it is the availability-first flow of §4.12 asking the candidate when they are free. Do not "fix" it by having the Main Agent guess a date. |
| 19 | **Never trust the advisor's slot list.** It calls the tool correctly and then still reports hours the tool never returned. `validate_slots()` re-reads each one from the database; keep it in the path, and read the `WARNING` it logs before blaming the schedule. |
