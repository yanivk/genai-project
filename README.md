<!-- PROJECT LOGO -->
<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" alt="Logo" width="120" height="120">
</p>

<h1 align="center">Recruiting Chatbot — Python Developer</h1>

<p align="center">
  A multi-agent SMS-style chatbot that screens candidates, answers their questions,<br>
  and books an interview — or closes the conversation politely.<br>
  <a href="#usage">Usage</a>
  ·
  <a href="#evaluation">Evaluation</a>
  ·
  <a href="#project-structure">Structure</a>
</p>

---
<br></br>

## Table of Contents

- [About The Project](#about-the-project)
- [Architecture](#architecture)
- [Features](#features)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Fine-Tuning](#fine-tuning)
- [Project Structure](#project-structure)
- [Deployment](#deployment)
- [To-Do List](#to-do-list)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---
<br></br>

## About The Project

> An SMS-based chatbot that interacts with job candidates for a **Python Developer**
> position. It gathers and verifies information, answers questions about the role, and
> ultimately schedules an interview with a human recruiter — or politely ends the
> conversation when the candidate is not interested.

The proof of concept runs in **Streamlit** instead of real SMS.

<div style="background: #272822; color: #f8f8f2; padding: 10px; border-radius: 8px;">
  <b>Technologies:</b> Python, LangChain, OpenAI API, Chroma, SQLite, Streamlit, scikit-learn
</div>

---
<br></br>

## Architecture

A **Main Agent** orchestrates each turn and consults three specialised **Advisors** before
deciding what to do.

```
candidate message
        │
        ▼
   Main Agent  ── receives and processes input
        │
        ├──────────────┬──────────────┐
        ▼              ▼              ▼
  Exit Advisor   Sched Advisor   Info Advisor
   (fine-tuned)     (SQL DB)      (Chroma DB)
        │              │              │
        └──────────────┴──────────────┘
                       ▼
                  Main Agent  ── decides and replies
```

| Component | Role | Backing resource |
|---|---|---|
| **Main Agent** | Runs the turn, owns the conversation memory, picks the action and writes the reply. | — |
| **Exit Advisor** | Judges whether the conversation should close, so uninterested candidates are not chased. **Fine-tuned** on the labeled dataset. | OpenAI fine-tuned model |
| **Scheduling Advisor** | Judges whether it is time to schedule; resolves relative dates ("next Friday") and proposes the 3 nearest free slots via function calling. | SQL database |
| **Info Advisor** | Answers questions about the role from the job description, and steers toward booking. | Chroma vector store |

Every turn resolves to exactly one action:

| Action | Meaning |
|---|---|
| `continue` | Keep the dialogue going — ask or answer a question. |
| `schedule` | Propose or renegotiate interview time slots. |
| `end` | The conversation is finished — the interview is confirmed, **or** the candidate opted out. |

---
<br></br>

## Features

- [x] Multi-agent orchestration (Main Agent + 3 Advisors)
- [x] LangChain agents, tools and conversation memory
- [x] Function calling against a SQL schedule database
- [x] RAG over the job description with a Chroma vector store
- [x] Fine-tuning pipeline for the Exit Advisor, with a graceful few-shot fallback
- [x] Streamlit chat UI
- [x] Evaluation on labeled real conversations (accuracy + confusion matrix)
- [ ] Deployment to Streamlit Community Cloud

---
<br></br>

## Getting Started

### Prerequisites

- Python >= 3.11
- An OpenAI API key

### Installation

```bash
git clone git@github.com:yanivk/genai-project.git
cd genai-project

python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env      # PowerShell: Copy-Item .env.example .env
```

Then fill in `OPENAI_API_KEY`. `.env` is gitignored — never commit it.

### Offline setup

Run these once, in order. They build the artifacts the app reads at runtime.

```bash
python scripts/seed_database.py             # -> data/tech.db
python scripts/build_vector_store.py        # -> data/chroma/ + data/vector_store.json
python scripts/run_finetuning.py --dry-run  # build the training JSONL, spend nothing
python scripts/run_finetuning.py            # optional; prints the fine-tuned model id
```

Both artifacts are already committed, so a fresh clone works without running the first
two. Re-run them only if you change the source data.

The fine-tuning step costs money and takes a few minutes; it asks for confirmation before
launching the job. Skip it if you like — the Exit Advisor falls back to the base model with
few-shot prompting until `FT_EXIT_ADVISOR_MODEL` is set.

---
<br></br>

## Usage

### Streamlit UI

```bash
streamlit run streamlit_app/streamlit_main.py
```

### Terminal

```bash
python -m app.main                                # talk to the bot as of today
python -m app.main --start 2026-07-15T15:12:00Z   # replay a dataset conversation
```

`--start` pins the conversation date, which is what relative expressions like *"next
Friday"* resolve against. The schedule runs to the end of 2027, so today always works.

Check your configuration without calling the API:

```bash
python -m app.main --check
```

### Code example

```python
from app.modules.main_agent.orchestrator import handle_turn

result = handle_turn(
    session_id="demo",
    candidate_message="I have three years' experience with Django and Flask.",
    conversation_start="2026-08-11T11:19:00Z",
)

print(result.action)   # 'continue'
print(result.message)  # the text sent to the candidate
```

---
<br></br>

## Evaluation

`data/sms_conversations.json` holds **15 real conversations / 103 turns**, of which 59
recruiter turns are labeled with the action the bot should have taken.

| Label | Count | Share |
|---|---|---|
| `continue` | 25 | 42.4% |
| `schedule` | 19 | 32.2% |
| `end` | 15 | 25.4% |

The task: given the history up to and including a candidate turn, predict the label of the
next recruiter turn.

Conversations span **2026-07-15 → 2026-08-11**. They were shifted from their original 2024
dates by a whole number of weeks, so every weekday — and therefore every relative reference
like *"next Tuesday"* — kept its meaning. Proposals falling on Monday were rewritten, since
the recruiter's schedule has no Monday or Saturday availability.

```bash
pytest tests/ -v                        # 63 offline tests, no API calls
jupyter notebook tests/test_evals.ipynb # full pipeline, reports the metrics below
```

### Results

| Split | n | Accuracy | Majority baseline | Lift |
|---|---|---|---|---|
| train (prompts tuned here) | 39 | **0.872** | 0.436 | +0.436 |
| test (held out) | 20 | **0.850** | 0.400 | +0.450 |

Per-class, on the held-out split:

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `continue` | 0.857 | 0.750 | 0.800 | 8 |
| `schedule` | 0.750 | 0.857 | 0.800 | 7 |
| `end` | **1.000** | **1.000** | **1.000** | 5 |

Train and test land within three points of each other, so the prompt calibration did not
overfit the conversations it was tuned on.

`end` — the class that covers both a confirmed booking and an opt-out — scores perfectly.
Getting there took three changes: teaching the Exit Advisor that a candidate naming *their
own* time counts as agreement; distinguishing that from a candidate *asking* for a slot,
which does not; and moving the action choice out of the LLM into a deterministic precedence
rule (`resolve_action()`), because the model reliably preferred `schedule` over `end`
whenever both advisors fired.

The residual `continue` ⇄ `schedule` errors are largely **label noise**, not a fixable defect.
The dataset gives opposite labels to near-identical situations — after *"I have three years'
experience with Django and Flask"* the recruiter asks another question (`continue`), while
after *"Yes, 3 years' experience"* they propose a time (`schedule`). No decision rule
satisfies both; the notebook prints the pair side by side.

Splitting happens at the **conversation** level, never the turn level — turns inside one
conversation share a history prefix, so a turn-level split would leak the test set into
prompt tuning and into the few-shot examples. The split is stratified by ending flavour so
both opt-outs and bookings appear on each side.

---
<br></br>

## Fine-Tuning

The Exit Advisor is the one component the brief asks to be **fine-tuned**, on OpenAI's API —
`gpt-4o-mini-2024-07-18` as the base.

```bash
python scripts/run_finetuning.py --dry-run   # build and inspect, spend nothing
python scripts/run_finetuning.py             # upload, launch, poll
```

A training row is literally an inference call: the same system prompt, the same fixed user
directive, and the target JSON as the assistant turn. Both halves are produced by calling the
advisor's own `build_system_text()` and `DIRECTIVE` rather than being re-typed, so training and
serving cannot drift apart.

Two details worth knowing:

- **The training split only.** The 5 held-out conversations never enter the JSONL — the same
  split function and seed the evaluation notebook uses. Training on all 15 and then reporting
  accuracy on 5 of them would look fine and mean nothing. A test asserts it.
- **The fine-tuned model gets a shorter prompt.** Replacing few-shot examples with learned
  behaviour is the point of fine-tuning, so `exit_advisor_finetuned.txt` drops the 7 worked
  examples and keeps the identity, the instructions and the output contract. Which file is used
  follows `FT_EXIT_ADVISOR_MODEL`.

The dry run reports what the job will learn from before you pay for it:

```
39 training rows
  should_end=True : 10  (booked 7, opted out 3)
  should_end=False: 29
```

That booked/opted-out breakdown is the line to check. `end` is terminal in **both** directions,
and a model trained on opt-outs alone learns that ending means rejection — wrong for 11 of the
15 conversations. `describe()` prints a warning if either flavour is missing.

**The fallback always works.** With `FT_EXIT_ADVISOR_MODEL` empty, the Exit Advisor runs on the
base model with few-shot prompting — which is how the evaluation results above were produced.
The app never hard-fails because a job is missing, expired, or still running.

---
<br></br>

## Project Structure

```text
genai-project/
├── CLAUDE.md                  Engineering guide: architecture, conventions, pitfalls
├── README.md
├── LICENSE
├── requirements.txt
├── .env.example
├── app/
│   ├── main.py                CLI entry point
│   ├── config.py              Env vars, model ids, paths — single source of truth
│   ├── prompts/               System prompts as .txt files
│   └── modules/
│       ├── main_agent/        Orchestrator, turn router, action literals
│       ├── advisors/          Exit / Scheduling / Info advisors + output schemas
│       ├── database/          SQLAlchemy engine, Schedule queries, SQLite seeder
│       ├── embedding/         PDF -> Chroma index, and retrieval
│       ├── finetuning/        JSONL builder, job launcher
│       └── evaluation/        Dataset loader, prediction runner, metrics
├── streamlit_app/
│   ├── streamlit_main.py      The Streamlit app
│   └── utils.py               Rendering helpers
├── scripts/
│   ├── seed_database.py
│   ├── build_vector_store.py
│   └── run_finetuning.py
├── tests/
│   ├── test_main.py
│   └── test_evals.ipynb       Accuracy + confusion matrix
└── data/
    ├── sms_conversations.json Labeled dataset
    ├── Python Developer Job Description.pdf
    ├── db_Tech.sql            Original T-SQL schema (reference)
    ├── tech.db                Generated SQLite database
    ├── vector_store.json      Chunks + embeddings seed (committed)
    └── chroma/                Persisted Chroma DB (generated, gitignored)
```

---
<br></br>

## Deployment

Deployed to **Streamlit Community Cloud**, pointing at `streamlit_app/streamlit_main.py`.

Two things make that work:

- **`data/tech.db` and `data/vector_store.json` are committed.** Streamlit Cloud cannot run
  the offline scripts, so those artifacts ship with the repo. That is also why the database
  is SQLite rather than SQL Server.
- **The Chroma database is rebuilt on first use.** Chroma pre-allocates its HNSW index for
  10,000 vectors and writes ~60 MB for our three chunks, so `data/chroma/` is gitignored and
  regenerated from the 94 KB JSON seed. Chroma is still what stores and answers every
  query — only the way it gets populated changes.
- **Secrets come from the Streamlit Secrets UI.** `app/config.py` reads `st.secrets` first
  and falls back to environment variables, so the same code runs in both places.

---
<br></br>

## To-Do List

- [x] Project scaffolding, conventions and configuration
- [x] SQLite seeder and schedule queries
- [x] Offline embedding pipeline (PDF → Chroma)
- [x] The three advisors
- [x] Main Agent orchestration
- [x] Streamlit chat wiring
- [x] Evaluation notebook results
- [x] Exit Advisor fine-tuning pipeline _(job not launched — runs on the few-shot fallback)_
- [ ] Cloud deployment

---
<br></br>

## License

Distributed under the Apache License 2.0. See `LICENSE` for more information.

---
<br></br>

## Acknowledgments

- [LangChain](https://python.langchain.com/)
- [OpenAI API](https://platform.openai.com/docs/overview)
- [Chroma](https://www.trychroma.com/)
- [Streamlit](https://streamlit.io/)
- [scikit-learn](https://scikit-learn.org/)

---
