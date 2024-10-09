from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

def chat(query: str):
    chat = ChatOpenAI(model_name="gpt-4o-mini", verbose=True, temperature=0)
    result = chat.invoke(input=query)
    return result.content