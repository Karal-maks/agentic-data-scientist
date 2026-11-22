import json, re
import pandas as pd
from app.llm import get_llm, ask

def critic_node(state):
    d, t, m = state["run_dir"], state["target"], state["metrics"]
    df = pd.read_csv(f"{d}/clean.csv")
    issues = []

    # only check columns the model actually used
    feats = m.get("features_used") or list(df.columns.drop(t))
    feats = [c for c in feats if c in df.columns and c != t]

    # 1. leakage: near-perfect correlation or ID-like features
    num = df[feats].select_dtypes("number")
    if t in df.select_dtypes("number"):
        for c in num.columns:
            v = abs(df[c].corr(df[t]))
            if v > 0.95:
                issues.append(f"Possible leakage: '{c}' has {v:.2f} correlation with target. Remove it.")
    for c in feats:
        if df[c].nunique() == len(df):
            issues.append(f"Feature '{c}' looks like an ID column. Remove it from the features.")

    # 2. overfitting and weak model
    try:
        cv = m["cv_scores"][m["best_model"]]
        test = m["test_metrics"][m["primary_metric"]]
        if abs(cv - test) > 0.10:
            issues.append(f"CV score {cv:.2f} and test score {test:.2f} differ a lot: possible overfitting.")
        if cv < m["baseline_cv"] + 0.05:
            issues.append("Best model barely beats the dummy baseline. Try better features or models.")
    except Exception as e:
        issues.append(f"metrics.json has the wrong structure ({e}). Follow the required keys exactly.")

    # 3. LLM code review
    prompt = f"""Review this ML code for methodology errors. Check: any fitting
done before the train/test split, test data used for model selection or tuning,
wrong metric for imbalanced data. Reply ONLY JSON: {{"problems": ["..."]}}
Code:\n{state['model_code']}"""
    raw = ask(get_llm(), prompt)
    try:
        issues += json.loads(re.search(r"\{.*\}", raw, re.S).group())["problems"]
    except Exception:
        pass

    return {"critic": {"passed": not issues, "issues": issues}, "feedback": issues}