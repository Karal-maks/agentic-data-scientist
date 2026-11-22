import sys, uuid, pathlib, time, html
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Autonomous Data Scientist", page_icon="🧪", layout="wide")
_css = (pathlib.Path(__file__).parent / "style.css").read_text(encoding="utf-8")
st.markdown(f"<style>{_css}</style>", unsafe_allow_html=True)

STEPS = [
    ("profile", "Profiler", "🔍", "Reading the data"),
    ("clean",   "Cleaner",  "🧹", "Fixing and tidying"),
    ("eda",     "Explorer", "📊", "Finding patterns"),
    ("model",   "Modeler",  "🤖", "Training models"),
    ("critic",  "Critic",   "🛡️", "Checking for mistakes"),
    ("report",  "Reporter", "📝", "Writing the report"),
]
ORDER = [s[0] for s in STEPS]


def stepper_html(status):
    cards = []
    for key, name, ico, desc in STEPS:
        s = status[key]
        bar = '<div class="bar"></div>' if s == "active" else ""
        cards.append(f'<div class="step {s}"><span class="ico">{ico}</span>'
                     f'<div class="name">{name}</div><div class="desc">{desc}</div>{bar}</div>')
    return '<div class="stepper">' + "".join(cards) + "</div>"


def feed_html(events):
    return '<div class="feed">' + "".join(f'<div class="feed-item">{e}</div>' for e in events) + "</div>"


def describe(node, m):
    if node == "profile":
        p = m["profile"]
        return f"🔍 Profiler: {p['task']} task, {p['rows']:,} rows × {p['cols']} columns"
    if node == "clean":
        first = (m.get("clean_log") or "done").strip().splitlines()[0][:110]
        return "🧹 Cleaner: " + html.escape(first)
    if node == "eda":
        return "📊 Explorer: key findings extracted and plots saved"
    if node == "model":
        mt = m["metrics"]
        score = mt["test_metrics"].get(mt["primary_metric"], float("nan"))
        return f"🤖 Modeler (round {m['iteration']}): {html.escape(str(mt['best_model']))}, test {mt['primary_metric']} = {score:.3f}"
    if node == "critic":
        c = m["critic"]
        if c["passed"]:
            return "🛡️ Critic: all checks passed"
        return f"🛡️ Critic: found {len(c['issues'])} issue(s), sending back for fixes"
    return "📝 Reporter: report is ready"


def next_node(node, m):
    if node == "critic":
        if m["critic"]["passed"] or m.get("iteration", 0) >= 3:
            return "report"
        return "model"
    i = ORDER.index(node)
    return ORDER[i + 1] if i + 1 < len(ORDER) else None


def demo_stream():
    metrics = {"best_model": "LogisticRegression", "primary_metric": "f1_macro",
               "test_metrics": {"f1_macro": 0.808}, "baseline_cv": 0.381,
               "cv_scores": {"LogisticRegression": 0.796}}
    seq = [
        ("profile", {"profile": {"task": "classification", "rows": 891, "cols": 11}, "iteration": 0}),
        ("clean", {"clean_log": "Dropped ID-like columns, stripped text, removed duplicates"}),
        ("eda", {"eda": "Survival rate 38.4%. Pclass is the strongest predictor."}),
        ("model", {"metrics": metrics, "iteration": 1}),
        ("critic", {"critic": {"passed": False, "issues": ["Example issue for the demo"]}}),
        ("model", {"metrics": metrics, "iteration": 2}),
        ("critic", {"critic": {"passed": True, "issues": []}}),
        ("report", {"report": "# Demo report\n\nThis is sample output shown in demo mode.\n\n"
                              "- Best model: Logistic Regression\n- Test macro F1: 0.808"}),
    ]
    for node, upd in seq:
        time.sleep(1.8)
        yield {node: upd}


# ---------------- page ----------------
st.markdown('<div class="hero"><h1>Autonomous Data Scientist</h1>'
            '<p>Upload a CSV and watch a team of AI agents profile, clean, model and report on it.</p></div>',
            unsafe_allow_html=True)

demo = st.toggle("Demo mode: preview the animation without using API quota")
f = st.file_uploader("Drop your CSV here", type="csv")

df, target = None, "Survived"
if f:
    df = pd.read_csv(f)
    st.dataframe(df.head())
    target = st.selectbox("Target column to predict", df.columns)

run = st.button("✨ Run analysis", disabled=not (demo or df is not None))

stepper_slot, feed_slot = st.empty(), st.empty()

if run:
    st.session_state.pop("result", None)
    status = {k: "pending" for k in ORDER}
    status["profile"] = "active"
    events, merged, run_dir = [], {}, None
    stepper_slot.markdown(stepper_html(status), unsafe_allow_html=True)
    try:
        if demo:
            stream = demo_stream()
        else:
            from app.graph import graph
            run_dir = pathlib.Path("runs") / uuid.uuid4().hex[:8]
            run_dir.mkdir(parents=True)
            (run_dir / "data.csv").write_bytes(f.getvalue())
            merged = {"run_dir": str(run_dir), "target": target}
            stream = graph.stream(dict(merged), stream_mode="updates")

        for chunk in stream:
            for node, upd in chunk.items():
                merged.update(upd)
                status[node] = "done"
                nxt = next_node(node, merged)
                if node == "critic" and nxt == "model":
                    status["model"], status["critic"] = "active", "pending"
                elif nxt:
                    status[nxt] = "active"
                events.append(describe(node, merged))
                stepper_slot.markdown(stepper_html(status), unsafe_allow_html=True)
                feed_slot.markdown(feed_html(events), unsafe_allow_html=True)
        st.session_state["result"] = {"m": merged, "run_dir": str(run_dir) if run_dir else None}
        st.toast("Analysis complete 🎉")
    except Exception as e:
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            st.error("The Gemini API quota is used up for now. Try again later, or use Demo mode to preview the UI.")
        else:
            st.error(f"Something went wrong: {msg[:300]}")
elif "result" in st.session_state:
    stepper_slot.markdown(stepper_html({k: "done" for k in ORDER}), unsafe_allow_html=True)

# ---------------- results ----------------
res = st.session_state.get("result")
if res:
    m = res["m"]
    mt, crit = m["metrics"], m["critic"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rounds used", m["iteration"])
    c2.metric("Critic", "Passed ✅" if crit["passed"] else "Open issues ⚠️")
    c3.metric("Test " + mt["primary_metric"], f"{mt['test_metrics'][mt['primary_metric']]:.3f}")
    c4.metric("Best model", str(mt["best_model"]))

    tab_report, tab_plots, tab_details = st.tabs(["📝 Report", "📈 Plots", "🧠 Agent details"])
    with tab_report:
        st.markdown(m.get("report", ""))
        st.download_button("⬇ Download report", m.get("report", ""), file_name="report.md")
    with tab_plots:
        plots = sorted(pathlib.Path(res["run_dir"], "plots").glob("*.png")) if res["run_dir"] else []
        if plots:
            cols = st.columns(2)
            for i, p in enumerate(plots):
                cols[i % 2].image(str(p))
        else:
            st.info("No plots to show (demo mode, or the agents saved none).")
    with tab_details:
        with st.expander("Cleaning log"):
            st.text(m.get("clean_log", ""))
        with st.expander("Exploration findings"):
            st.text(m.get("eda", ""))
        with st.expander("Critic issues"):
            st.write(crit["issues"] or "No open issues.")
        with st.expander("All metrics"):
            st.json(mt)