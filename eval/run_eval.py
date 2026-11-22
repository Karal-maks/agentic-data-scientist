import os, time, pathlib, shutil
import pandas as pd
from sklearn import datasets
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from app.graph import graph

LEAK_TESTS = 2      # how many datasets also get the injected-leakage test
PAUSE_SECONDS = 60  # pause between runs to respect free-tier rate limits


def load_datasets():
    out = {}
    for name, loader in [("iris", datasets.load_iris), ("wine", datasets.load_wine)]:
        out[name] = (loader(as_frame=True).frame, "target")
    for f in sorted(pathlib.Path("eval/datasets").glob("*__*.csv")):
        name, target = f.stem.split("__", 1)
        out[name] = (pd.read_csv(f), target)
    only = os.getenv("EVAL_ONLY")
    if only:
        out = {k: v for k, v in out.items() if k in only.split(",")}
    return out


def human_baseline(df, target):
    """A plain hand-written pipeline on the same 80/20 split."""
    df = df.dropna(subset=[target])
    X, y = df.drop(columns=[target]), df[target]
    X = X.loc[:, [c for c in X.columns if X[c].nunique() < len(X)]]  # drop ID-like columns
    task = "classification" if (y.nunique() <= 15 or y.dtype == "object") else "regression"
    num = X.select_dtypes("number").columns
    cat = X.columns.difference(num)
    pre = ColumnTransformer([
        ("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
        ("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                        ("o", OneHotEncoder(handle_unknown="ignore"))]), cat)])
    est = RandomForestClassifier(random_state=42) if task == "classification" else RandomForestRegressor(random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42,
                                          stratify=y if task == "classification" else None)
    pipe = Pipeline([("pre", pre), ("m", est)]).fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    return f1_score(yte, pred, average="macro") if task == "classification" else r2_score(yte, pred)


def run_one(name, df, target, leak=False):
    run_dir = pathlib.Path("runs") / f"eval_{name}{'_leak' if leak else ''}"
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    d = df.copy()
    if leak:
        d["leak_feature"] = d[target]
    d.to_csv(run_dir / "data.csv", index=False)
    t0 = time.time()
    row = {"dataset": name, "leak_test": leak}
    try:
        res = graph.invoke({"run_dir": str(run_dir), "target": target})
        m = res["metrics"]
        row.update(ok=True, rounds=res["iteration"], critic_passed=res["critic"]["passed"],
                   agent_score=m["test_metrics"][m["primary_metric"]],
                   leak_removed="leak_feature" not in m.get("features_used", []))
    except Exception as e:
        row.update(ok=False, error=str(e)[:200])
    row["seconds"] = round(time.time() - t0)
    return row


def main():
    rows, leak_done = [], 0
    for name, (df, target) in load_datasets().items():
        print(f"\n=== {name} ===")
        r = run_one(name, df, target)
        try:
            r["human_score"] = human_baseline(df, target)
        except Exception as e:
            r["human_score"] = None
            print("baseline failed:", e)
        rows.append(r)
        print(r)
        pd.DataFrame(rows).to_csv("eval/results.csv", index=False)
        time.sleep(PAUSE_SECONDS)

        if leak_done < LEAK_TESTS:
            leak_done += 1
            lr = run_one(name, df, target, leak=True)
            rows.append(lr)
            print(lr)
            pd.DataFrame(rows).to_csv("eval/results.csv", index=False)
            time.sleep(PAUSE_SECONDS)

    res = pd.DataFrame(rows)
    normal, leak = res[~res["leak_test"]], res[res["leak_test"]]
    print("\n===== SUMMARY =====")
    print(f"Success rate: {normal['ok'].mean():.0%} ({int(normal['ok'].sum())}/{len(normal)})")
    good = normal[normal["ok"] & normal["human_score"].notna()]
    if len(good):
        print(f"Avg agent score: {good['agent_score'].mean():.3f} | Avg human baseline: {good['human_score'].mean():.3f}")
        print(f"Agent as % of human: {(good['agent_score'] / good['human_score']).mean():.0%}")
        print(f"Avg time per run: {good['seconds'].mean():.0f}s")
    if len(leak) and leak["ok"].any():
        print(f"Leakage removed in {int(leak['leak_removed'].sum())}/{int(leak['ok'].sum())} injected-leak runs")


if __name__ == "__main__":
    main()