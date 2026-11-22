from app.agents.profiler import profile_node
from app.agents.cleaner import clean_node, eda_node

state = {"run_dir": "runs/test1", "target": "Survived"}
state.update(profile_node(state))
print("Task type:", state["profile"]["task"])

state.update(clean_node(state))
print("CLEANING:\n", state["clean_log"])

state.update(eda_node(state))
print("EDA:\n", state["eda"])