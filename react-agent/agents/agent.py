from typing import Any, Dict, List
from langchain.prompts.prompt import PromptTemplate
# from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
# from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from tools.addition import Addition
from tools.subtraction import Subtraction
from tools.multiplication import Multiplication
from tools.division import Division
from tools.vdb_tools import RAGTool
from langchain.tools.render import render_text_description


from dotenv import load_dotenv

load_dotenv()


def talk(question: str, chat_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    llm = ChatOpenAI(temperature=0.35, model_name="gpt-4o-mini")
    # react_prompt = hub.pull("hwchase17/react")
    
    agent_tools = [       
        Addition().get_tool_definition(),
        Subtraction().get_tool_definition(),
        Multiplication().get_tool_definition(),
        Division().get_tool_definition(),
        RAGTool().get_tool_definition()
    ]

    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}
    The tools can be used as many times as needed to answer the question. You have the liberty to change the parameters as needed to better use the tools.

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Previous conversation history:
    {chat_history}
    
    Question: {input}
    Thought: {agent_scratchpad}
"""

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(agent_tools), 
        tool_names=", ".join([t.name for t in agent_tools])
    )
    
    agent = create_react_agent(
        llm=llm, tools=agent_tools, prompt=prompt
    )
    agent_executor = AgentExecutor(
        agent=agent, tools=agent_tools, verbose=True, return_intermediate_steps=True, handle_parsing_errors=True, max_execution_time=120, max_iterations=120
    )

    print(chat_history.append)
    result = agent_executor.invoke(
        input={
            "input": question,
            "chat_history": chat_history
        }
    )
    
    return result

if __name__ == "__main__":
    print("Hello React Langchain")
    print(talk( "how much is (2 + 1 * 2)/2 - 1?", None))

