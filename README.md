# Autonomous Data Scientist

A multi-agent AI system that takes any tabular CSV file and autonomously profiles, cleans, explores, models and reports on it. A **critic agent** reviews every result for data leakage, overfitting and weak models, and sends the work back for fixes before a report is written.

> Built with LangGraph, Gemini, scikit-learn, Docker and Streamlit.

## Demo

<!-- Add a screenshot or GIF of the Streamlit app here -->
![App screenshot](docs/screenshot.png)

## How it works

```
Profiler -> Cleaner -> EDA -> Modeler -> Critic --+-- issues found --> back to Modeler (max 3 rounds)
                                                  +-- passed --------> Reporter
```

| Agent | What it does |
|---|---|
| Profiler | Reads the CSV, detects the task type (classification or regression), summarises types, missing values and target distribution |
| Cleaner | Writes and runs code to fix types, strip text, drop duplicates, constant and ID-like columns |
| EDA | Creates plots and extracts the key findings with real numbers |
| Modeler | Builds leakage-safe scikit-learn pipelines, compares models with cross-validation, evaluates once on a held-out test set |
| Critic | Checks for leakage, ID-like features, overfitting and weak models, and reviews the modelling code for methodology errors |
| Reporter | Writes a plain-language report for a non-technical reader |

## Key design decisions

- **Sandboxed code execution.** All LLM-written code runs inside a Docker container with no network access and limited memory and CPU. It never runs directly on the host machine.
- **Self-repair loop.** If generated code fails, the error is sent back to the model to fix it (up to 3 attempts).
- **Leakage-safe preprocessing.** The cleaner is not allowed to impute, scale or encode. All learned preprocessing happens inside a cross-validated pipeline, fitted on training data only.
- **Rule-based plus LLM critic.** Leakage and overfitting checks are plain code (cheap and deterministic); an LLM review of the modelling code is added on top.
- **A false-positive fix.** The first critic version flagged columns the model never used and wasted retry rounds. It now checks only the features actually used in training.

## Results

> Fill these in from `eval/results.csv` after running the evaluation. Do not publish numbers you have not measured.

| Metric | Value |
|---|---|
| Datasets evaluated | N |
| Pipeline success rate | X / N |
| Agent score vs hand-written baseline (same split) | X% of baseline |
| Injected-leakage cases caught | X / Y |
| Average time per run | X s |

The baseline is a plain hand-written Random Forest pipeline evaluated on the same 80/20 split.

## Known limitations

- Results vary between runs, because the LLM writes new code each time.
- Evaluation size was limited by free-tier API quota.
- Tabular data only (CSV). No time-series-specific handling.
- The task type is detected with a simple rule (few unique values or text target = classification), which can be wrong for unusual targets.

## Project structure

```
app/
  agents/        profiler, cleaner (+EDA), modeler, critic, reporter
  graph.py       LangGraph workflow
  llm.py         Gemini client with pacing and retry
  sandbox.py     Docker sandbox runner
  state.py       shared state
ui/streamlit_app.py
eval/run_eval.py evaluation script
Dockerfile.sandbox
```

## How to run (Windows PowerShell)

Requirements: Python 3.11+, Docker Desktop, a Gemini API key from Google AI Studio.

```powershell
git clone https://github.com/Karal-maks/agentic-data-scientist.git
cd agentic-data-scientist
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# add your API key (do not commit this file)
Set-Content .env "GOOGLE_API_KEY=your_key_here"

# build the sandbox image (Docker Desktop must be running)
docker build -f Dockerfile.sandbox -t ds-sandbox .

# launch the app
$env:PYTHONPATH = "."
streamlit run ui\streamlit_app.py
```

Upload a CSV, choose the target column, and click **Run analysis**.

## Run the evaluation

```powershell
$env:PYTHONPATH = "."
python eval\run_eval.py
```

Optional settings: `EVAL_ONLY=iris,wine` to choose datasets, `CRITIC_LLM=0` to skip the LLM code review, `LLM_MODEL` to change the model.

## Author

Karal Maks, [LinkedIn](https://linkedin.com/in/KARALMAKS), [GitHub](https://github.com/Karal-maks)