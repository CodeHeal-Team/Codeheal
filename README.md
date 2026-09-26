# CodeHeal — AI-Powered Automated Bug Diagnosis & Repair

CodeHeal is an AI-powered debugging and code repair platform built for the IBM watsonx Hackathon. It helps developers identify bugs, understand their root causes, generate reproducible tests, apply code fixes, and verify those fixes through automated testing.

## Live Demo

* **Frontend:** https://codeheal-frontend.vercel.app
* **Backend API:** https://codeheal-backend.onrender.com
* **API Documentation:** https://codeheal-backend.onrender.com/docs

> Note: The backend runs on Render's free tier and may take some time to wake up after a period of inactivity.

## The Problem

Debugging software can be time-consuming. Developers often need to reproduce a failure, inspect logs, identify the faulty code, write a test, implement a fix, and rerun the test suite manually.

## Our Solution

CodeHeal brings these steps together in an automated debugging pipeline. It uses IBM watsonx.ai to assist with root-cause diagnosis and code repair, while automated tests help verify the result.

## Key Features

* Repository and sample-project loading
* Automated execution of the initial test suite
* AI-assisted root-cause diagnosis
* Identification of the affected source file
* Generation of a reproducible test
* Automated patch generation and application
* Patch validation
* Test execution after the repair
* Pipeline progress and results displayed in the frontend

## How It Works

1. **Load:** Load the selected sample project.
2. **Test:** Run the initial test suite to identify a failure.
3. **Diagnose:** Analyze the failure and identify the likely root cause.
4. **Generate a test:** Create a reproducible test for the bug.
5. **Repair:** Generate, validate, and apply a code patch.
6. **Verify:** Rerun the test to check whether the fix resolves the failure.
7. **Report:** Display the pipeline results and code changes.

## Demo Scenarios

The live demo includes three seeded bug scenarios. Each starts with a failing test, uses IBM watsonx.ai to assist with diagnosis and repair, generates a reproducible regression test, applies a patch, and reruns tests to verify the result.

### 1. Missing `None` validation (`none_bug`)

A calculator function attempts division without validating a `None` input. The initial test exposes the resulting `TypeError`; CodeHeal diagnoses the missing guard, applies a fix, and verifies the regression test.

### 2. Broken API routing (`broken_api`)

A route lookup fails because a key contains a visually similar Cyrillic character instead of the expected ASCII character. CodeHeal identifies the mismatched key, applies a repair, and verifies the route with a regression test.

### 3. Off-by-one slice boundary (`off_by_one`)

The `last_n(items, n)` utility calculates its slice start as `len(items) - n`. When `n` exceeds the list length, the negative start index returns an incorrect slice. CodeHeal diagnoses the missing clamp, changes the start to `max(0, len(items) - n)`, and verifies the fix with tests.

## Technology Stack

| Component         | Technology                        |
| ----------------- | --------------------------------- |
| Frontend          | React, Vite, JavaScript           |
| Backend API       | Python, FastAPI                   |
| AI                | IBM watsonx.ai, IBM Granite model |
| Automated testing | pytest                            |
| Frontend hosting  | Vercel                            |
| Backend hosting   | Render                            |

## Architecture

```text
User
 |
 v
React + Vite Frontend (Vercel)
 |
 v
FastAPI Backend (Render)
 |
 +--> Sample repository and test execution
 |
 +--> IBM watsonx.ai for AI-assisted diagnosis and repair
 |
 +--> Patch validation and post-fix testing
 |
 v
Pipeline results displayed in the frontend
```

## Repository Structure

```text
Codeheal/
├── backend/
│   ├── api/
│   ├── core/
│   ├── samples/
│   ├── tests/
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   ├── package.json
│   └── .env.example
├── .env.example
├── .gitignore
└── README.md
```

## Run Locally

### Prerequisites

* Python installed
* Node.js and npm installed
* IBM watsonx.ai credentials for running AI-powered backend operations

### 1. Clone the repository

```bash
git clone https://github.com/CodeHeal-Team/Codeheal.git
cd Codeheal
```

### 2. Configure the backend

From the repository root, run these commands in PowerShell:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `backend/.env` file using `backend/.env.example` as a template. Fill in your own IBM watsonx.ai credentials:

```dotenv
WATSONX_API_KEY=your_watsonx_api_key
WATSONX_PROJECT_ID=your_watsonx_project_id
WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

Never commit this file or share your API key.

Start the backend from the `backend` directory:

```bash
uvicorn api.app:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Interactive API documentation is available at `http://localhost:8000/docs`.

### 3. Configure the frontend

Open a second terminal from the repository root:

```bash
cd frontend
npm install
```

Create a `frontend/.env` file containing:

```dotenv
VITE_API_BASE_URL=http://localhost:8000
```

Start the frontend:

```bash
npm run dev
```

Open the local URL printed by Vite, normally `http://localhost:5173`.

## Environment Variables

| Variable             | Purpose                                  |
| -------------------- | ---------------------------------------- |
| `WATSONX_API_KEY`    | Authenticates requests to IBM watsonx.ai |
| `WATSONX_PROJECT_ID` | Identifies the watsonx project           |
| `WATSONX_URL`        | IBM watsonx.ai service endpoint          |
| `VITE_API_BASE_URL`  | Backend API URL used by the frontend     |

Use the relevant `.env.example` files as templates. Never put real credentials in source code, commit them to Git, or include them in public project archives.

## Security

* API keys and credentials must be supplied through environment variables.
* Real `.env` files must remain untracked and private.
* Only placeholder values should appear in `.env.example` files.
* Review changes before committing or pushing to GitHub.

## Team

**Project:** CodeHeal
**Organization:** CodeHeal-Team
**Repository:** https://github.com/CodeHeal-Team/Codeheal

## Future Scope

Potential improvements include supporting more programming languages, integrating additional test frameworks, improving patch evaluation, and adding richer debugging reports and repository integrations.

---

Built for the IBM watsonx Hackathon.

## IBM Bob 2.0 Development Evidence

The following screenshots document the CodeHeal development process and live application. The Bob task-session and validation screenshots provide development evidence; the live-application screenshot shows the deployed frontend.

1. [Backend pipeline task/session summary](docs/bob-evidence/01-backend-pipeline.png)
2. [Response parser task summary](docs/bob-evidence/02-response-parser-summary.png)
3. [Module validation and test results](docs/bob-evidence/03-validation-and-tests.png)
4. [Live application frontend](docs/bob-evidence/04-live-application.png)

See the [evidence folder guide](docs/bob-evidence/README.md) for context. Screenshots should be genuine, readable, and free of API keys, tokens, or other secrets.
