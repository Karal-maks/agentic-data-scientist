import json
from app.agents.base import generate_and_run

def model_node(state):
    t, task = state["target"], state["profile"]["task"]
    fb = "\n".join(state.get("feedback", [])) or "none"
    prompt = f"""Read /work/clean.csv. Target '{t}', task: {task}.
Requirements:
- train_test_split FIRST (80/20, random_state=42, stratify if classification)
- sklearn Pipeline + ColumnTransformer (impute, scale numeric, one-hot categorical,
  handle_unknown='ignore'), fit on train only
- baseline: DummyClassifier / DummyRegressor
- candidates: LogisticRegression or Ridge, RandomForest, XGBoost
- 5-fold CV on train only; pick best by CV; evaluate ONCE on test
- primary_metric must be higher-is-better: f1_macro (classification) or r2 (regression)
- write /work/metrics.json with keys: baseline_cv (number), cv_scores (dict model name -> mean score),
  best_model (string, must be a key of cv_scores), primary_metric (string),
  test_metrics (dict with the primary_metric as a key), features_used (list)
- all numbers in metrics.json must be plain Python floats
- save /work/plots/feature_importance.png and /work/model.joblib
Fix these reviewer issues from the previous round: {fb}"""
    code, _ = generate_and_run(prompt, state["run_dir"])
    metrics = json.load(open(f"{state['run_dir']}/metrics.json"))
    return {"model_code": code, "metrics": metrics,
            "iteration": state["iteration"] + 1}