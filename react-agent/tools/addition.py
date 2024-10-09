from langchain_core.tools import Tool

class Addition:
    
    def __init__(self) -> None:
        pass
    
    def add(self, a, b):
        return a + b

    def get_tool_definition(self):
        return Tool(
            name="Addition",
            func=lambda x: str(self.add(*map(float, x.split(',')))),
            description="Adds two numbers. Input should be two numbers separated by a comma."
        )
    
addition = Addition()