import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(override=True)

def get_llm():
    return ChatGoogleGenerativeAI(
        model=os.getenv("LLM_MODEL", "gemini-3.6-flash"),
        temperature=0,
    )

def ask(llm, messages) -> str:
    """Return the reply as plain text, whatever format the model sends back."""
    content = llm.invoke(messages).content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return content