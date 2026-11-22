import re, time
from app.llm import get_llm, ask
from app.sandbox import run_code

SYSTEM = ("You are a senior data scientist. Reply with ONE python code block only. "
          "Files are in /work. Use only pandas, numpy, scikit-learn, xgboost, "
          "matplotlib, seaborn, joblib. Print concise results.")

def extract_code(text: str) -> str:
    m = re.search(r"```python(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()

def generate_and_run(prompt: str, run_dir: str, max_tries: int = 3):
    llm = get_llm()
    msgs = [("system", SYSTEM), ("user", prompt)]
    err = ""
    for _ in range(max_tries):
        code = extract_code(ask(llm, msgs))
        ok, out, err = run_code(code, run_dir)
        if ok:
            return code, out
        msgs += [("assistant", f"```python\n{code}\n```"),
                 ("user", f"It failed with:\n{err}\nFix the code.")]
        time.sleep(3)
    raise RuntimeError(f"Agent failed after {max_tries} tries: {err}")
