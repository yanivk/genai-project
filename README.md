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
- [x] Fine-tuned Exit Advisor, with a graceful few-shot fallback
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
python scripts/seed_database.py         # -> data/tech.db
python scripts/build_vector_store.py    # -> data/chroma/
python scripts/run_finetuning.py        # optional; prints the fine-tuned model id
```

The fine-tuning step costs money and takes a few minutes. Skip it if you like — the Exit
Advisor falls back to the base model with few-shot prompting until
`FT_EXIT_ADVISOR_MODEL` is set.

---
<br></br>

## Usage

### Streamlit UI

```bash
streamlit run streamlit_app/streamlit_main.py
```

### Terminal

```bash
python -m app.main --start 2024-04-03T15:12:00Z
```

Pass `--start` to pin the conversation date. Relative expressions like *"next Friday"*
resolve against it, and the schedule database only covers 2024 — without it the bot
anchors on today and finds no slots.

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
    conversation_start="2024-04-30T11:19:00Z",
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

```bash
pytest tests/ -v
jupyter notebook tests/test_evals.ipynb
```

The notebook reports accuracy against the **majority-class baseline (42.4%)**, a confusion
matrix, per-class precision/recall, and a table of every misclassified turn.

Splitting happens at the **conversation** level, never the turn level — turns inside one
conversation share a history prefix, so a turn-level split would leak the test set into
training and into the few-shot examples.

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
    └── chroma/                Persisted vector index
```

---
<br></br>

## Deployment

Deployed to **Streamlit Community Cloud**, pointing at `streamlit_app/streamlit_main.py`.

Two things make that work:

- **`data/tech.db` and `data/chroma/` are committed.** Streamlit Cloud cannot run the
  offline scripts, so the artifacts have to ship with the repo. That is also why the
  database is SQLite rather than SQL Server.
- **Secrets come from the Streamlit Secrets UI.** `app/config.py` reads `st.secrets` first
  and falls back to environment variables, so the same code runs in both places.

---
<br></br>

## To-Do List

- [x] Project scaffolding, conventions and configuration
- [ ] SQLite seeder and schedule queries
- [ ] Offline embedding pipeline
- [ ] The three advisors
- [ ] Main Agent orchestration
- [ ] Exit Advisor fine-tuning
- [ ] Streamlit chat wiring
- [ ] Evaluation notebook results
- [ ] Cloud deployment

---
<br></br>

## License

Distributed under the MIT License. See `LICENSE` for more information.

---
<br></br>

## Acknowledgments

- [LangChain](https://python.langchain.com/)
- [OpenAI API](https://platform.openai.com/docs/overview)
- [Chroma](https://www.trychroma.com/)
- [Streamlit](https://streamlit.io/)
- [scikit-learn](https://scikit-learn.org/)

---
