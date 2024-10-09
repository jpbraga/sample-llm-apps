from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from typing import Any, Dict, List
from langchain import hub
from langchain_openai import ChatOpenAI

load_dotenv()

def chat(query: str, chat_history: List[Dict[str, Any]] = []):
    chat = ChatOpenAI(model_name="gpt-4o-mini", verbose=True, temperature=0)



    template = """
    {input}

    Chat History:
    {chat_history}
"""

    prompt = PromptTemplate.from_template(template=template).format(
        input=query,
        chat_history=chat_history
    )
    
    result = chat.invoke(input=prompt)
    print(result.content)
    return result.content