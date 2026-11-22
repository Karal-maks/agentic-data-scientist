import os, time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(override=True)
_last_call = 0.0
MIN_GAP = float(os.getenv("MIN_GAP_SECONDS", "13"))

def get_llm():
    return ChatGoogleGenerativeAI(
        model=os.getenv("LLM_MODEL", "gemini-3.6-flash"),
    )

def ask(llm, messages) -> str:
    """Return the reply as plain text; pace calls and retry if rate-limited."""
    global _last_call
    for attempt in range(6):
        wait_gap = MIN_GAP - (time.time() - _last_call)
        if wait_gap > 0:
            time.sleep(wait_gap)
        _last_call = time.time()
        try:
            content = llm.invoke(messages).content
            break
        except Exception as e:
            msg = str(e)
            if ("429" in msg or "RESOURCE_EXHAUSTED" in msg) and attempt < 5:
                if "per day" in msg.lower() or "PerDay" in msg:
                    raise  # daily limit: waiting won't help
                wait = 30 * (attempt + 1)
                print(f"Rate limited. Waiting {wait}s (attempt {attempt + 1}/6)...")
                time.sleep(wait)
                continue
            raise
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return content