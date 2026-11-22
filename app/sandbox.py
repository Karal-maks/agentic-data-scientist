import subprocess, pathlib

def run_code(code: str, workdir: str, timeout: int = 180):
    wd = pathlib.Path(workdir).resolve()
    (wd / "script.py").write_text(code, encoding="utf-8")
    cmd = ["docker", "run", "--rm", "--network", "none", "--memory", "1g",
           "--cpus", "1", "--pids-limit", "128",
           "-v", f"{wd}:/work", "-w", "/work", "ds-sandbox", "python", "script.py"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode == 0, p.stdout[-4000:], p.stderr[-4000:]
    except subprocess.TimeoutExpired:
        return False, "", "Timeout: code ran too long"