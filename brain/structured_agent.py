from typing import Dict, Any, List
from chat.message import MessageType, m
from chat.service import Service as ChatService
from tools.tools import _TOOL_FUNCS, TOOLS_SCHEMA
from brain.final_response import FinalResponse
from pydantic import BaseModel, Field
import json

# Initialize LLM service lazily to avoid import-time API key issues
_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatService("gpt-4o-mini")
    return _llm

# Reactive thinking step model (similar to old implementation)
_reactive_thinking_prompt = (
    "You are thinking out loud before deciding whether to use a tool. "
    "You will be given a conversation history between a user and an assistant. "
    "Your goal is to analyze this conversation and determine what single next action will best move toward fully answering the user's request.\n\n"
    "First, review the conversation history to understand:\n"
    "- What is the user's original request?\n"
    "- What information has already been gathered?\n"
    "- What progress has been made so far?\n"
    "- What specific details or context are mentioned (like categories, numbers, etc.)?\n\n"
    "Then assess whether the user's original request has already been completely satisfied. "
    "If not, think about what specific piece of information is still missing.\n\n"
    "Finally, decide:\n"
    "- Should you call a tool to get that missing information?\n"
    "- Or do you already have everything needed and should just proceed to respond?\n\n"
    "IMPORTANT GUIDELINES:\n"
    "1. If you have all the information needed, set use_tool=False and provide a clear next_step that summarizes what you will say in your final response.\n"
    "2. If you need multiple pieces of information, prefer to gather them one at a time. This helps maintain clarity and makes it easier to track progress.\n"
    "3. For follow-up questions like 'show me 2 examples of the category', understand what 'the category' refers to from context.\n"
    "4. Be specific about what tool you need and why - don't be vague.\n"
    "5. If the user mentions specific numbers (like '2 examples'), make sure to use those in your tool parameters.\n\n"
    "Respond using the fields:\n"
    "- 'use_tool': true if a tool is needed, false otherwise\n"
    "- 'reasoning': a detailed explanation of your analysis, what you found in the conversation, and why you made this decision\n"
    "- 'next_step': a clear, specific description of the immediate next action—either call a specific tool with parameters or proceed to respond"
)

class ReactiveThinkingStep(BaseModel):
    reasoning: str = Field(description="A brief explanation of whether a tool call is needed or not, and why.")
    use_tool: bool = Field(description="True if a tool should be called next, False if no tool is needed.")
    next_step: str = Field(description="A single clear sentence describing the immediate next action—either call a specific tool or proceed without tools.")

def _get_structured_prompt() -> str:
    """Get the system prompt for structured queries."""
    tools_doc = _generate_tool_documentation(_TOOL_FUNCS)
    
    # Get dataset info for context
    dataset_info_func, _ = _TOOL_FUNCS["dataset_info"]
    dataset_info = dataset_info_func()
    
    return (
        "You are a structured data agent specialized in the Bitext Customer Support Service dataset. "
        "You handle direct, specific queries that require precise data retrieval.\n"
        "\nYou have access to the following tools:\n"
        f"\n{tools_doc}\n\n"
        f"Here is useful dataset info to help you decide which tool to use:\n"
        f"\n{dataset_info}\n\n"
        "For structured queries, focus on:\n"
        "- dataset_info: For general dataset statistics and information\n"
        "- data_slicer: For filtering, grouping, and sampling data\n"
        "- exact_search: For finding specific text or patterns\n"
        "- calculator: For numerical calculations\n"
        "\nExplain your reasoning when deciding which tool to use.\n"
        "Use tools when needed to answer the user's question accurately.\n"
        "If you have access to a calculator tool, use it for all calculations, even simple ones.\n"
    )

def _generate_tool_documentation(tools_dict: Dict) -> str:
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

def _think_next_step(messages: List[Dict[str, Any]]) -> ReactiveThinkingStep:
    """Analyze conversation and decide what to do next (reactive thinking)."""
    llm = _get_llm()
    
    # Prepare messages for thinking
    thinking_messages = [
        {"role": "system", "content": _reactive_thinking_prompt}
    ]
    
    # Add conversation history (only user-facing messages for context)
    for msg in messages:
        if msg.get("message_type") == MessageType.USER_FACING:
            thinking_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    
    # Get thinking step decision
    response = llm.chat(
        thinking_messages,
        response_format=ReactiveThinkingStep
    )
    
    thinking_step = response.choices[0].message.parsed
    
    # Enhanced console output for thinking step
    print(f"   📋 CONVERSATION ANALYSIS:")
    print(f"      - User messages analyzed: {len([m for m in messages if m.get('role') == 'user'])}")
    print(f"      - Assistant responses: {len([m for m in messages if m.get('role') == 'assistant'])}")
    print(f"      - Tool results: {len([m for m in messages if m.get('message_type') == MessageType.TOOL_RESULT])}")
    
    return thinking_step

def _execute_tool(name: str, args: Dict[str, Any]):
    """Execute a tool with the given arguments."""
    # Get the tool function and schema
    func, schema = _TOOL_FUNCS[name]
    
    # Check for required parameters
    required_params = schema["parameters"].get("required", [])
    missing_params = [param for param in required_params if param not in args]
    
    if missing_params:
        return {
            "error": f"Missing required parameters: {', '.join(missing_params)}",
            "required_parameters": required_params,
            "provided_parameters": list(args.keys())
        }
        
    # Execute the tool
    try:
        return func(**args)
    except Exception as e:
        return {
            "error": str(e),
            "tool": name,
            "args": args
        }

def structured_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle structured queries using reactive thinking - step by step until satisfied.
    """
    user_message = state["user_message"]
    messages = state.get("messages", [])
    
    print(f"\n{'='*60}")
    print(f"📊 STRUCTURED AGENT: Processing '{user_message}'")
    print(f"{'='*60}")
    
    # Track thinking messages for UI display
    thinking_messages = []
    
    try:
        # Working copy of messages for the reactive loop
        working_messages = messages.copy()
        
        # Add the current user message
        user_msg = m(role="user", content=user_message, message_type=MessageType.USER_FACING)
        working_messages.append(user_msg)
        
        step_count = 0
        
        # Reactive thinking loop - continue until request is satisfied
        while True:
            step_count += 1
            print(f"\n{'─'*50}")
            print(f"🔄 STEP {step_count}: Thinking about next action...")
            print(f"{'─'*50}")
            
            # Think about what to do next
            thinking_step = _think_next_step(working_messages)
            
            # Add thinking message with enhanced formatting
            thinking_msg = m(
                role="assistant",
                content=thinking_step.reasoning,  # Show detailed reasoning as main content
                reasoning=thinking_step.next_step,  # Put brief action in reasoning field
                message_type=MessageType.THINKING
            )
            working_messages.append(thinking_msg)
            thinking_messages.append(thinking_msg)
            
            print(f"\n💭 THINKING:")
            print(f"   Reasoning: {thinking_msg['reasoning']}")
            print(f"   Next Step: {thinking_msg['content']}")
            print(f"   Use Tool: {'Yes' if thinking_step.use_tool else 'No'}")
            
            # If no tool needed, we're ready to respond
            if not thinking_step.use_tool:
                print(f"\n✅ REQUEST SATISFIED: Ready to generate final response")
                break
            
            print(f"\n🔧 TOOL EXECUTION PHASE:")
            print(f"   {'─'*40}")
            
            # Get tool calls for this step
            llm = _get_llm()
            resp = llm.chat(working_messages, tools_json=TOOLS_SCHEMA)
            msg = resp.choices[0].message
            
            if msg.tool_calls:
                # Add tool call message with enhanced reasoning
                tool_names = [tc.function.name for tc in msg.tool_calls]
                tool_call_msg = m(
                    role="assistant",
                    content=f"I'll use the {', '.join(tool_names)} tool(s) to gather the specific information needed.",
                    reasoning=msg.content or f"Based on my analysis, I need to gather some specific data to provide you with the information you're looking for",
                    message_type=MessageType.TOOL_CALL,
                    tool_calls=[
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                )
                working_messages.append(tool_call_msg)
                thinking_messages.append(tool_call_msg)
                
                print(f"   🎯 Tool Selection: {', '.join(tool_names)}")
                print(f"   📝 Reasoning: {tool_call_msg['reasoning']}")
                print(f"   {'─'*40}")
                
                # Execute tools and collect results
                for i, tc in enumerate(msg.tool_calls, 1):
                    name = tc.function.name
                    args = json.loads(tc.function.arguments or "{}")
                    
                    print(f"\n   🛠️  TOOL {i}: {name}")
                    print(f"      Parameters: {args}")
                    print(f"      {'─'*30}")
                    
                    result = _execute_tool(name, args)
                    
                    # Add tool result message
                    tool_result_msg = m(
                        role="tool",
                        content=json.dumps(result, ensure_ascii=False),
                        message_type=MessageType.TOOL_RESULT,
                        reasoning=f"Tool {name} executed successfully with the provided parameters",
                        tool_call_id=tc.id
                    )
                    working_messages.append(tool_result_msg)
                    thinking_messages.append(tool_result_msg)
                    
                    # Print tool execution summary with better formatting
                    print(f"      ✅ Execution completed")
                    if isinstance(result, list):
                        print(f"      📊 Results: {len(result)} items returned")
                    elif isinstance(result, dict):
                        if 'count' in result:
                            print(f"      📊 Results: {result['count']} matches found")
                        else:
                            print(f"      📊 Results: {len(result)} key-value pairs returned")
                    else:
                        print(f"      📊 Results: Data retrieved successfully")
                    print(f"      {'─'*30}")
                
                print(f"\n🔄 Continuing to next thinking step...")
                continue
            
            # No tool calls but still need tools - this shouldn't happen
            print(f"\n⚠️  WARNING: Expected tool calls but none were made")
            break
        
        print(f"\n{'='*60}")
        print(f"🎯 GENERATING FINAL RESPONSE")
        print(f"{'='*60}")
        
        # Generate final response
        llm = _get_llm()
        final_resp = llm.chat(working_messages, response_format=FinalResponse)
        final_response = final_resp.choices[0].message.parsed
        
        print(f"\n📝 FINAL RESPONSE:")
        print(f"   Content: {final_response.content}")
        print(f"   Reasoning: {final_response.reasoning}")
        print(f"{'='*60}")
        
        response = m(
            role="assistant",
            content=final_response.content,
            reasoning=final_response.reasoning,
            message_type=MessageType.USER_FACING
        )
        
    except Exception as e:
        print(f"\n❌ ERROR in structured agent: {e}")
        print(f"{'='*60}")
        response = m(
            role="assistant",
            content="I encountered an error while processing your structured query. Please try again.",
            reasoning=f"Error: {str(e)}",
            message_type=MessageType.USER_FACING
        )
    
    # Always append to the existing message history
    return {
        **state,
        "messages": state["messages"] + thinking_messages + [response],
        "final_answer": response["content"],
        "current_step": "processed_structured_with_reactive_thinking",
        "thinking_messages": thinking_messages
    } 