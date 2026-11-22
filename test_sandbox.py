import os
from app.sandbox import run_code

os.makedirs("runs/t", exist_ok=True)
print(run_code('print("hello")', "runs/t"))