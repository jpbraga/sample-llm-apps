from langchain_core.tools import Tool

class Subtraction:
    
    def __init__(self) -> None:
        pass

    def subtract(self, a, b):
        return a - b

    def get_tool_definition(self):
        return Tool(
            name="Subtraction",
            func=lambda x: str(self.subtract(*map(float, x.split(',')))),
            description="Subtracts second number from first. Input should be two numbers separated by a comma."
        )