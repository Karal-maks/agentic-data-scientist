from app.llm import get_llm, ask
print(ask(get_llm(), "Say hello in one short sentence."))
