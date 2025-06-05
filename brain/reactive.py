from typing import List, Dict, Tuple
from .strategy import Strategy
from message_models.message import MessageType, m
from tools.tools import TOOLS_SCHEMA
from .reactive_thinking_step import ReactiveThinkingStep, _system_prompt as _thinking_system_prompt

_reactive_instructions = (
    "When answering questions:\n"
    "1. Focus on gathering information one step at a time\n"
    "2. For comparative analysis, consider both quantitative and qualitative aspects\n"
    "3. Build up your answer gradually using the information gathered\n"
    "4. Keep track of what information you've already collected\n"
    "5. Format your final response clearly and concisely\n"
)

class Reactive(Strategy):

    def get_system_prompt(self) -> str:
        return f"{self._get_base_prompt()}\n\n{_reactive_instructions}"

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

        # convert the first system message to user message if it exists
        modified_messages = messages.copy()
        if modified_messages and modified_messages[0]["role"] == "system":
            modified_messages[0] = {
                "role": "user",
                "content": modified_messages[0]["content"]
            }

        # add new system message at the beginning
        modified_messages.insert(0, {"role": "system", "content": _thinking_system_prompt})

        response = self._llm.chat(
            modified_messages,
            tools_json=None,
            response_format=ReactiveThinkingStep,
        )
        return response.choices[0].message.parsed