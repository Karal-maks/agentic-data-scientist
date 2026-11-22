from app.graph import graph

res = graph.invoke({"run_dir": "runs/test1", "target": "Survived"})
print("ROUNDS USED:", res["iteration"])
print("METRICS:", res["metrics"])
print("CRITIC:", res["critic"])
print("\nREPORT:\n", res["report"])