import numexpr as ne

def calculator(expression: str) -> dict:
    try:
        result = ne.evaluate(expression)
        return {
            "result": float(result),
            "expression": expression
        }
    except Exception as e:
        return {
            "error": str(e),
            "expression": expression
        }
    
TOOL_FUNC = {
    "calculator": (
        calculator,
        {
            "name": "calculator",
            "description": (
                "Evaluate mathematical expressions. Supports basic arithmetic, trigonometric functions, "
                "logarithms, and other common mathematical operations. "
                "Use this tool when you need to perform mathematical calculations or when the user asks for numerical computations. "
                "Always prefer using this tool over doing calculations yourself, even for simple arithmetic. "
                "Provide the expression as a string (e.g., '2 + 2', 'sin(pi/2)', 'sqrt(16)')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate. Examples: '2 + 2', 'sin(pi/2)', 'sqrt(16)', 'log(100)', '6095/100'"
                    }
                },
                "required": ["expression"]
            }
        }
    )
}