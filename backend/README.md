# CodeHeal

**AI-powered code testing and self-healing developer tool**
IBM Bob 2.0 Hackathon · Python prototype

CodeHeal demonstrates a closed self-healing loop powered by IBM Granite on watsonx.ai:

```
repository → run tests → detect failure → diagnose root cause
→ generate reproducible failing test → apply targeted fix
→ rerun test → verify fix → display complete process in a dashboard
```

---

## Architecture

| Module | Location | Responsibility |
|--------|----------|----------------|
| Core Brain | `core/` | Loads repo, runs tests, owns pipeline state, validates agent outputs |
| Agent Intelligence | `agents/` | Diagnostic, Test Generator, and Refactoring agents |
| Command Center | `dashboard/` | Streamlit UI — drives Module 1, displays every stage |

---

## Setup

### 1. Prerequisites

- Python 3.11+
- `git` available on `PATH`

### 2. Install dependencies

```bash
cd codeheal
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your IBM Cloud credentials
```

Required variables:

| Variable | Description |
|----------|-------------|
| `WATSONX_API_KEY` | Your IBM Cloud API key |
| `WATSONX_PROJECT_ID` | Your watsonx.ai project ID |
| `WATSONX_URL` | watsonx.ai service URL (e.g. `https://us-south.ml.cloud.ibm.com`) |

### 4. Run the dashboard

```bash
streamlit run dashboard/app.py
```

Open <http://localhost:8501> in your browser.

---

## Demo Scenarios

Three sample projects are bundled under `samples/`. Each has a deliberate bug and an existing test that fails:

| Scenario | Bug Type | Location |
|----------|----------|----------|
| `none_bug` | `None` input crashes a function | `samples/none_bug/` |
| `broken_api` | Misspelled route key causes `KeyError` | `samples/broken_api/` |
| `off_by_one` | Fencepost error in a loop/slice | `samples/off_by_one/` |

**Recommended demo order:** `none_bug` → `off_by_one` → `broken_api`

---

## Running a pipeline from the command line

```bash
cd codeheal
python -m core.pipeline samples/none_bug
```

---

## Resetting a sample between runs

Each sample is a git repository. The dashboard "Reset" button runs:

```bash
git checkout .
```

inside the sample directory to restore all files to their committed state.

---

## Project structure

```
codeheal/
├── agents/
│   ├── base_agent.py              # shared watsonx.ai client and prompt helper
│   ├── diagnostic_agent.py
│   ├── test_generator_agent.py
│   ├── refactoring_agent.py
│   └── placeholders.py            # stub agents used during development
├── core/
│   ├── pipeline.py                # orchestrator
│   ├── runner.py                  # subprocess test runner
│   ├── repo_loader.py             # reads source files
│   ├── models.py                  # all shared data types
│   └── response_parser.py         # JSON/diff validator with retry logic
├── dashboard/
│   └── app.py                     # Streamlit UI
├── samples/
│   ├── none_bug/
│   ├── broken_api/
│   └── off_by_one/
├── .env.example
├── requirements.txt
└── README.md
```
