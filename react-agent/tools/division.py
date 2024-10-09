from langchain_core.tools import Tool

class Division:
    
    def __init__(self) -> None:
        pass

    def divide(self, a, b):
        if b == 0:
            return "Error: Division by zero is undefined."
        return a / b

    def get_tool_definition(self):
        return Tool(
            name="Division",
            func=lambda x: str(self.divide(*map(float, x.split(',')))),
            description="Divides first number by second. Input should be two numbers separated by a comma."
        )
