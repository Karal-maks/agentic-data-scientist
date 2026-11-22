import json
from app.agents.base import generate_and_run

def clean_node(state):
    prompt = f"""Profile: {json.dumps(state['profile'], default=str)}
Target: {state['target']}
Write code that reads /work/data.csv and writes /work/clean.csv.
ALLOWED: fix dtypes, parse dates, strip strings, drop duplicate rows,
drop ID-like or constant columns, drop rows with missing target.
NOT ALLOWED: imputing, scaling, encoding, or anything that learns from
data statistics (that happens later inside a CV pipeline to avoid leakage).
Print a short list of what you changed."""
    _, out = generate_and_run(prompt, state["run_dir"])
    return {"clean_log": out}

def eda_node(state):
    prompt = f"""Read /work/clean.csv (target: {state['target']}).
Create /work/plots/ and save 4-6 useful plots (target distribution,
missingness, top correlations, key feature vs target) as PNG.
Print the 8 most important findings as bullet points with numbers."""
    _, out = generate_and_run(prompt, state["run_dir"])
    return {"eda": out}