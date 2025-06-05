from typing import List, Dict, Tuple
from .strategy import Strategy
from message_models.message import MessageType, m
from tools.tools import TOOLS_SCHEMA
from .reactive_thinking_step import ReactiveThinkingStep

_system_prompt = (
    "You are thinking out loud before deciding whether to use a tool. "
    "You will be given a conversation history between a user and an assistant. "
    "Your goal is to analyze this conversation and determine what single next action will best move toward fully answering the user's request.\n\n"
    "First, review the conversation history to understand:\n"
    "- What is the user's original request?\n"
    "- What information has already been gathered?\n"
    "- What progress has been made so far?\n\n"
    "Then assess whether the user's original request has already been completely satisfied. "
    "If not, think about what specific piece of information is still missing.\n\n"
    "Finally, decide:\n"
    "- Should you call a tool to get that missing information?\n"
    "- Or do you already have everything needed and should just proceed to respond?\n\n"
    "IMPORTANT GUIDELINES:\n"
    "1. If you have all the information needed, set use_tool=False and provide a clear next_step that summarizes what you will say in your final response.\n"
    "2. If you need multiple pieces of information, prefer to gather them one at a time. This helps maintain clarity and makes it easier to track progress.\n"
    "Respond using the fields:\n"
    "- 'use_tool': true if a tool is needed, false otherwise\n"
    "- 'reasoning': a short explanation of your decision, referencing the conversation history\n"
    "- 'next_step': a one-sentence description of the next action"
)

class Reactive(Strategy):

    def think(self, messages: List[Dict[str, str]]) -> Tuple[Dict[str, str], List[Dict[str, str]]]:

        working = messages.copy()
        new_msgs: List[Dict[str, str]] = []

        while True:
            thinking_step = self._think_next_step(working)
            thinking_msg = m(
                role="assistant",
                content=thinking_step.next_step,
                reasoning=thinking_step.reasoning,
                message_type=MessageType.THINKING
            )
            working.append(thinking_msg)
            new_msgs.append(thinking_msg)

            print(f"\n{thinking_msg['reasoning']}")
            print(f"My next step should be: {thinking_msg['content']}")

            if not thinking_step.use_tool:
                answer, _ = self._final_response(working)
                return answer, new_msgs

            # Now get the tool calls - request only one tool
            resp = self._llm.chat(
                working, 
                tools_json=TOOLS_SCHEMA
            )
            msg = resp.choices[0].message

            if msg.tool_calls:
                working, new_msgs = self._handle_tool_calls(msg, working, new_msgs)
                continue

            # no tool calls → final answer
            answer, _ = self._final_response(working)
            
            return answer, new_msgs
        
    def _think_next_step(self, messages: list[dict]) -> ReactiveThinkingStep:

        # Convert the first system message to user message if it exists
        modified_messages = messages.copy()
        if modified_messages and modified_messages[0]["role"] == "system":
            modified_messages[0] = {
                "role": "user",
                "content": modified_messages[0]["content"]
            }

        # Add new system message at the beginning
        modified_messages.insert(0, {"role": "system", "content": _system_prompt})

        response = self._llm.chat(
            modified_messages,
            tools_json=None,
            response_format=ReactiveThinkingStep,
        )
        return response.choices[0].message.parsed