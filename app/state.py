from typing import TypedDict, List

class State(TypedDict, total=False):
    run_dir: str
    target: str
    profile: dict
    clean_log: str
    eda: str
    metrics: dict
    model_code: str
    critic: dict
    feedback: List[str]
    iteration: int
    report: str