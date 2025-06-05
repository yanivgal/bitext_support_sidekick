from typing import List, Dict, Tuple
from .strategy import Strategy
from .planning_thinking import PlanningThinking
from message_models.message import MessageType, m
from tools.tools import TOOLS_SCHEMA
from .planning_thinking import _system_prompt as _thinking_system_prompt

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

class Plan(Strategy):

    def get_system_prompt(self) -> str:
        return f"{self._get_base_prompt()}\n\n{_planning_instructions}"

    def think(self, messages: List[Dict[str, str]]) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
        
        plan = self._plan_thinking(messages)
        
        working = messages.copy()
        new_msgs = []
        
        plan_msg = m(
            role="assistant",
            content=f"{plan.goal}\n\nSteps:\n" + "\n".join(
                f"{i+1}. {step.action}\n   Expected: {step.expected_result}\n   Reasoning: {step.reasoning}" + 
                (f"\n   Depends on: {[d + 1 for d in step.depends_on]}" if step.depends_on else "")
                for i, step in enumerate(plan.steps)
            ),
            reasoning=f"Planning to achieve: {plan.goal}",
            message_type=MessageType.THINKING
        )
        working.append(plan_msg)
        new_msgs.append(plan_msg)
        
        print(f"\n📋 The Plan:\n{plan.goal}")
        for i, step in enumerate(plan.steps):
            print(f"   Step {i+1}:")
            print(f"      Action: {step.action}")
            print(f"      Expected: {step.expected_result}")
            print(f"      Reasoning: {step.reasoning}")
            if step.depends_on:
                print(f"      Depends on: {[d + 1 for d in step.depends_on]}")
        
        print("\nExecuting the plan...")
        
        for i, step in enumerate(plan.steps):
            step_msg = m(
                role="assistant",
                content=step.action,
                reasoning=step.reasoning,
                message_type=MessageType.THINKING
            )
            working.append(step_msg)
            new_msgs.append(step_msg)
            
            print(f"\n📝 Step {i+1}: {step.reasoning}")
            
            if "tool" in step.action.lower():
                resp = self._llm.chat(working, tools_json=TOOLS_SCHEMA)
                msg = resp.choices[0].message
                
                if msg.tool_calls:
                    working, new_msgs = self._handle_tool_calls(msg, working, new_msgs)
        
        answer, _ = self._final_response(working)
        
        return {
            "content": answer["content"],
            "reasoning": answer["reasoning"]
        }, new_msgs
    
    def _plan_thinking(self, messages: list[dict]) -> PlanningThinking:
        
        response = self._llm.chat(
            messages + [{
                "role": "system",
                "content": _thinking_system_prompt
            }],
            tools_json=None,
            response_format=PlanningThinking
        )
        return response.choices[0].message.parsed