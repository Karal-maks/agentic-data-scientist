from langgraph.graph import StateGraph, END
from app.state import State
from app.agents.profiler import profile_node
from app.agents.cleaner import clean_node, eda_node
from app.agents.modeler import model_node
from app.agents.critic import critic_node
from app.agents.reporter import report_node

def route(state):
    return "done" if state["critic"]["passed"] or state["iteration"] >= 3 else "retry"

g = StateGraph(State)
for name, fn in [("profile", profile_node), ("clean", clean_node), ("eda", eda_node),
                 ("model", model_node), ("critic", critic_node), ("report", report_node)]:
    g.add_node(name, fn)
g.set_entry_point("profile")
g.add_edge("profile", "clean")
g.add_edge("clean", "eda")
g.add_edge("eda", "model")
g.add_edge("model", "critic")
g.add_conditional_edges("critic", route, {"retry": "model", "done": "report"})
g.add_edge("report", END)
graph = g.compile()