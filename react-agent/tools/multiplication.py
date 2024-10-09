from langchain_core.tools import Tool

class Multiplication:
    
    def __init__(self) -> None:
        pass
        
    def multiply(self, a, b):
        return a * b

    def get_tool_definition(self):
        return Tool(
            name="Multiplication",
            func=lambda x: str(self.multiply(*map(float, x.split(',')))),
            description="Multiplies two numbers. Input should be two numbers separated by a comma."
        )
