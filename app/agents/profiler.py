import pandas as pd

def profile_node(state):
    df = pd.read_csv(f"{state['run_dir']}/data.csv")
    t = state["target"]
    task = "classification" if (df[t].nunique() <= 15 or df[t].dtype == "object") else "regression"
    prof = {"rows": len(df), "cols": df.shape[1],
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing": df.isna().sum().to_dict(),
            "nunique": df.nunique().to_dict(),
            "task": task, "head": df.head(5).to_dict("records")}
    if task == "classification":
        prof["target_dist"] = df[t].value_counts(normalize=True).round(3).to_dict()
    return {"profile": prof, "iteration": 0, "feedback": []}
