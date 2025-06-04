from typing import Dict, List
from tools.tools import _TOOL_FUNCS

_reactive_instructions = (
            "When answering questions:\n"
            "1. Focus on gathering information one step at a time\n"
            "2. For comparative analysis, consider both quantitative and qualitative aspects\n"
            "3. Build up your answer gradually using the information gathered\n"
            "4. Keep track of what information you've already collected\n"
            "5. Format your final response clearly and concisely\n"
        )
        
_planning_instructions = (
            "You are a planning agent. Your job is to create a structured plan to answer the user's question. "
            "The plan should include:\n"
            "1. A sequence of steps, each with:\n"
            "   - What action needs to be taken\n"
            "   - What result we expect to get\n"
            "   - Why this step is needed and how it contributes to the goal\n"
            "   - Any dependencies on previous steps\n"
            "2. A clear overall goal\n\n"
            "CRITICAL: Category names MUST be exact matches from the available categories list. "
            "For example, if the user asks about 'account issues', you must use 'ACCOUNT' as the category name. "
            "Never use variations or different cases - only use the exact category names as shown in the available categories list."
        )

def get_system_prompt(mode: str, capabilities: Dict[str, List[str]]) -> str:
        base_prompt = get_base_prompt(capabilities)
        if mode == "reactive":
            system_prompt = f"{base_prompt}\n\n{_reactive_instructions}"
        else:
            system_prompt = f"{base_prompt}\n\n{_planning_instructions}"
        return system_prompt


def get_base_prompt(dataset_info: Dict[str, List[str]]) -> str:
     tools_doc = generate_tool_documentation(_TOOL_FUNCS)
     
     return (
          "You are the best data agent specialized in the Bitext Customer Support Service dataset. "
            "You answer questions about the dataset and its contents clearly and concisely.\n"
            "\nYou have access to the following tools:\n"
            f"\n{tools_doc}\n\n"
            f"Here is useful dataset info to help you decide which tool to use:\n"
            f"\n{dataset_info}\n\n"
            "\nExplain your reasoning when deciding which tool to use.\n"
            "Use tools when needed to answer the user's question accurately.\n"
            "If you have access to a calculator tool, use it for all calculations, even simple ones.\n"
            "If a tool is expected to take a long time to run, inform the user as early as possible about the expected duration.\n"
        )

def generate_tool_documentation(tools_dict: Dict) -> str:
    """Generate documentation for tools from the _TOOL_FUNCS dictionary."""
    docs = []
    for name, (_, schema) in tools_dict.items():
        # Get the description
        description = schema["description"]
        
        # Get parameters if they exist
        params = schema.get("parameters", {}).get("properties", {})
        param_docs = []
        for param_name, param_info in params.items():
            param_desc = param_info.get("description", "")
            param_type = param_info.get("type", "")
            param_docs.append(f"  - {param_name} ({param_type}): {param_desc}")
        
        # Format the tool documentation
        tool_doc = f"- {name}: {description}"
        if param_docs:
            tool_doc += "\n  Parameters:\n" + "\n".join(param_docs)
        docs.append(tool_doc)
    
    return "\n".join(docs)