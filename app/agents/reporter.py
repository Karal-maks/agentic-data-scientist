import json
from app.llm import get_llm, ask

def report_node(state):
    prompt = f"""Write a clear markdown report for a non-technical manager.
Sections: Summary, Data overview, Key findings, Model results (compare to baseline),
Risks and limitations, Recommended next steps.
Profile: {json.dumps(state['profile'], default=str)[:3000]}
Cleaning: {state['clean_log']}
EDA: {state['eda']}
Metrics: {json.dumps(state['metrics'])}
Reviewer issues still open: {state['critic']['issues']}
Rounds used: {state['iteration']}"""
    report = ask(get_llm(), prompt)
    open(f"{state['run_dir']}/report.md", "w", encoding="utf-8").write(report)
    return {"report": report}