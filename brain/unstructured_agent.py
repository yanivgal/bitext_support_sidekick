from typing import Dict, Any, List
from chat.message import MessageType, m
from chat.service import Service as ChatService
from tools.tools import _TOOL_FUNCS, TOOLS_SCHEMA
from brain.final_response import FinalResponse
import json

# Instantiate the LLM service
_llm = ChatService("gpt-4o-mini")

def _get_unstructured_prompt() -> str:
    """Get the system prompt for unstructured queries."""
    tools_doc = _generate_tool_documentation(_TOOL_FUNCS)
    
    # Get dataset info for context
    dataset_info_func, _ = _TOOL_FUNCS["dataset_info"]
    dataset_info = dataset_info_func()
    
    return (
        "You are an unstructured analysis agent specialized in the Bitext Customer Support Service dataset. "
        "You handle analysis, summary, and pattern discovery queries that require deeper insights.\n"
        "\nYou have access to the following tools:\n"
        f"\n{tools_doc}\n\n"
        f"Here is useful dataset info to help you decide which tool to use:\n"
        f"\n{dataset_info}\n\n"
        "For unstructured queries, focus on:\n"
        "- semantic_search: For finding conceptually similar content and patterns\n"
        "- find_common_questions: For discovering frequent question patterns\n"
        "- aggregator: For complex aggregations and analysis\n"
        "- data_slicer: For filtering and grouping data for analysis\n"
        "\nSPECIAL HANDLING FOR CAPABILITY QUESTIONS:\n"
        "If the user asks 'What can you do?' or similar capability questions:\n"
        "- Explain your capabilities in a conversational, friendly way\n"
        "- Mention the types of queries you can handle (structured vs unstructured)\n"
        "- Give examples of what kinds of questions work well\n"
        "- Suggest that they can ask for specific recommendations later\n"
        "- Be encouraging and helpful, not just list features\n"
        "\n"
        "SPECIAL HANDLING FOR SUGGESTION QUESTIONS:\n"
        "If the user asks 'What do you think I should ask next?' or 'Why do you think that?' or similar:\n"
        "- Have a conversation about what they might want to explore\n"
        "- Explain your reasoning based on their interests and past queries\n"
        "- Give thoughtful suggestions with explanations\n"
        "- Be conversational and encouraging\n"
        "- Don't just list suggestions - explain why they might be interesting\n"
        "- Ask follow-up questions to understand their interests better\n"
        "\nExplain your reasoning when deciding which tool to use.\n"
        "Use tools when needed to provide comprehensive analysis and insights.\n"
        "Focus on providing meaningful summaries and discovering patterns in the data.\n"
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

def unstructured_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle unstructured queries using LLM to intelligently select and call analysis tools.
    """
    user_message = state["user_message"]
    messages = state.get("messages", [])
    
    print(f"🔍 Unstructured Agent: Processing '{user_message}'")
    
    # Prepare messages for the LLM
    llm_messages = [
        {"role": "system", "content": _get_unstructured_prompt()}
    ]
    
    # Add conversation history
    for msg in messages:
        if msg["message_type"] == MessageType.USER_FACING:
            llm_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    
    # Add the current user message
    llm_messages.append({
        "role": "user",
        "content": user_message
    })
    
    # Track thinking messages for UI display
    thinking_messages = []
    
    try:
        # Add initial thinking message with better reasoning
        thinking_msg = m(
            role="assistant",
            content="I need to understand what you're asking and figure out the best way to help you with this question.",
            reasoning="Starting to analyze your question to determine the best approach for providing helpful insights and information",
            message_type=MessageType.THINKING
        )
        thinking_messages.append(thinking_msg)
        print(f"\n{thinking_msg['reasoning']}")
        print(f"My next step should be: {thinking_msg['content']}")
        
        # Get tool calls from LLM
        resp = _llm.chat(llm_messages, tools_json=TOOLS_SCHEMA)
        msg = resp.choices[0].message
        
        if msg.tool_calls:
            # Add tool call thinking message with better reasoning
            tool_names = [tc.function.name for tc in msg.tool_calls]
            tool_call_msg = m(
                role="assistant",
                content=f"I'll use the {', '.join(tool_names)} tool(s) to analyze the data and discover patterns.",
                reasoning=msg.content or f"Based on your question, I need to gather some information to provide you with the best possible answer",
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
            thinking_messages.append(tool_call_msg)
            
            print(f"\n🔧 {tool_call_msg['reasoning']}\n")
            print(f"🔧 Taking actions to gather the required information...\n")
            
            # Execute tools and collect results
            tool_results = []
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments or "{}")
                print(f"   🛠️  Executing tool: {name} with args: {args}")
                
                result = _execute_tool(name, args)
                tool_results.append({
                    "tool": name,
                    "args": args,
                    "result": result
                })
                
                # Add tool result message
                tool_result_msg = m(
                    role="tool",
                    content=json.dumps(result, ensure_ascii=False),
                    message_type=MessageType.TOOL_RESULT,
                    reasoning=f"Tool {name} executed with args: {args}",
                    tool_call_id=tc.id
                )
                thinking_messages.append(tool_result_msg)
                
                # Print tool execution summary
                if isinstance(result, list):
                    print(f"   ✅ {name} returned {len(result)} items")
                elif isinstance(result, dict):
                    if 'count' in result:
                        print(f"   ✅ {name} found {result['count']} matches")
                    else:
                        print(f"   ✅ {name} returned {len(result)} key-value pairs")
                else:
                    print(f"   ✅ {name} execution completed")
            
            # Add final thinking message with better reasoning
            final_thinking_msg = m(
                role="assistant",
                content="Now I have the information I need. Let me organize this into a clear, helpful response for you.",
                reasoning="I've gathered the relevant data. Now I need to put it all together into a comprehensive answer that addresses your question clearly and usefully.",
                message_type=MessageType.THINKING
            )
            thinking_messages.append(final_thinking_msg)
            print(f"\n{final_thinking_msg['reasoning']}")
            print(f"My next step should be: {final_thinking_msg['content']}")
            
            # Generate final response with tool results
            final_messages = llm_messages + [
                {
                    "role": "assistant",
                    "content": "I've analyzed the data and gathered insights.",
                    "tool_calls": [
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
                }
            ]
            
            # Add tool results
            for i, tool_result in enumerate(tool_results):
                final_messages.append({
                    "role": "tool",
                    "content": json.dumps(tool_result["result"], ensure_ascii=False),
                    "tool_call_id": msg.tool_calls[i].id
                })
            
            # Generate final response
            final_resp = _llm.chat(final_messages, response_format=FinalResponse)
            final_response = final_resp.choices[0].message.parsed
            
            print(f"🔍 Final response generated: {final_response.content}")
            print(f"🔍 Final reasoning: {final_response.reasoning}")
            
            response = m(
                role="assistant",
                content=final_response.content,
                reasoning=final_response.reasoning,
                message_type=MessageType.USER_FACING
            )
            
        else:
            # No tool calls, generate direct response
            final_resp = _llm.chat(llm_messages, response_format=FinalResponse)
            final_response = final_resp.choices[0].message.parsed
            
            response = m(
                role="assistant",
                content=final_response.content,
                reasoning=final_response.reasoning,
                message_type=MessageType.USER_FACING
            )
    
    except Exception as e:
        print(f"Error in unstructured agent: {e}")
        response = m(
            role="assistant",
            content="I encountered an error while processing your analysis query. Please try again.",
            reasoning=f"Error: {str(e)}",
            message_type=MessageType.USER_FACING
        )
    
    # Always append to the existing message history
    return {
        **state,
        "messages": state["messages"] + thinking_messages + [response],
        "final_answer": response["content"],
        "current_step": "processed_unstructured_with_tools",
        "thinking_messages": thinking_messages
    } 