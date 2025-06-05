from abc import abstractmethod
from typing import Any,List, Dict, Tuple
from chat_service import ChatService
from .final_response import FinalResponse
import json
from message_models.message import Message, MessageType, m
from tools.tools import _TOOL_FUNCS, TOOLS_SCHEMA

class Strategy:

    def __init__(self, llm: ChatService):
        self._llm = llm

    @abstractmethod
    def think(self, messages: List[Dict[str, str]]) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
        pass

    def _final_response(self, messages: List[Dict[str, str]]) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
        
        resp = self._llm.chat(
            messages,
            tools_json=None,
            response_format=FinalResponse
        )
        final_response = resp.choices[0].message.parsed
        answer = {
            "content": final_response.content,
            "reasoning": final_response.reasoning
        }

        print(f"\n{answer['reasoning']}")
        
        return answer, []
    
    def _handle_tool_calls(
        self, 
        msg, 
        working: List[Dict[str, str]], 
        new_msgs: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        
        assistant_msg = m(
            role="assistant",
            content=msg.content.strip() if msg.content else "Taking actions to gather the required information...",
            reasoning=msg.content.strip() if msg.content else "Taking actions to gather the required information...",
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
        working.append(assistant_msg)
        new_msgs.append(assistant_msg)
        
        print(f"\n🔧 {assistant_msg['reasoning']}\n")

        # run each tool, append its reply
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            print(f"   🛠️  Executing tool: {name} with args: {args}")
            result = self.execute_tool(name, args)

            tool_msg = m(
                role="tool",
                content=json.dumps(result, ensure_ascii=False),
                message_type=MessageType.TOOL_RESULT,
                reasoning=f"Tool {name} executed with args: {args}",
                tool_call_id=tc.id
            )
            working.append(tool_msg)
            new_msgs.append(tool_msg)
            
            # Print tool execution
            result = json.loads(tool_msg['content'])
            if isinstance(result, list):
                print(f"   ✅ {name} returned {len(result)} items")
            elif isinstance(result, dict):
                if 'count' in result:
                    print(f"   ✅ {name} found {result['count']} matches")
                else:
                    print(f"   ✅ {name} returned {len(result)} key-value pairs")
            else:
                print(f"   ✅ {name} execution completed")

        return working, new_msgs
    
    def execute_tool(self, name: str, args: Dict[str, Any]):
        """Execute a tool with the given arguments.
        
        Args:
            name: Name of the tool to execute
            args: Dictionary of arguments for the tool
            
        Returns:
            The result of the tool execution
        """
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
            
        # Normalize category name if present in args
        if "category" in args:
            args["category"] = self._normalize_category(args["category"])
            
        # Execute the tool
        try:
            return func(**args)
        except Exception as e:
            return {
                "error": str(e),
                "tool": name,
                "args": args
            }