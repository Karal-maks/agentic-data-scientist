import sys, uuid, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st
from app.graph import graph

st.set_page_config(page_title="Autonomous Data Scientist", layout="wide")
st.title("Autonomous Data Scientist")
st.caption("Upload a CSV. A team of AI agents profiles, cleans, models and reports on it.")

f = st.file_uploader("Upload a CSV", type="csv")
if f:
    df = pd.read_csv(f)
    st.dataframe(df.head())
    target = st.selectbox("Target column to predict", df.columns)
    if st.button("Run analysis"):
        run_dir = pathlib.Path("runs") / uuid.uuid4().hex[:8]
        run_dir.mkdir(parents=True)
        (run_dir / "data.csv").write_bytes(f.getvalue())
        with st.spinner("Agents are working... this can take a few minutes"):
            res = graph.invoke({"run_dir": str(run_dir), "target": target})

        c1, c2, c3 = st.columns(3)
        c1.metric("Rounds used", res["iteration"])
        c2.metric("Critic passed", "Yes" if res["critic"]["passed"] else "No")
        m = res["metrics"]
        c3.metric("Test " + m["primary_metric"], f"{m['test_metrics'][m['primary_metric']]:.3f}")

        if res["critic"]["issues"]:
            st.warning("Open reviewer issues: " + "; ".join(res["critic"]["issues"]))

        st.markdown(res["report"])
        for p in sorted((run_dir / "plots").glob("*.png")):
            st.image(str(p))