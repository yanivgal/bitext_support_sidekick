from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from tools.tools import _TOOL_FUNCS, TOOLS_SCHEMA
from message_models.reactive_thinking_step import ReactiveThinkingStep
from message_models.final_response import FinalResponse
from message_models.message import Message, MessageType, m
from message_models.planning_thinking import PlanningThinking, PlanningStep
from system_prompts import get_system_prompt
from chat_service import ChatService
from scope_checker.checker import Checker
from scope_checker.scope import ScopeEnum
from thinking.reactive import Reactive
from thinking.plan import Plan

class Agent:

    def __init__(
        self,
        model: str = "gpt-4o-mini",  # replace with Qwen2.5-32B if desired
        mode: str = "reactive",  # Add mode parameter
    ):
        if mode not in ["reactive", "plan"]:
            raise ValueError("mode must be 'reactive' or 'plan'")
        
        self._mode = mode
        self._model = model
        self._capabilities = self._discover_capabilities()
        self._system_prompt = get_system_prompt(self._mode, self._capabilities)
        self._llm = ChatService(model)
        self._scope = Checker(model)
        self._brain = Reactive(self._llm) if mode == "reactive" else Plan(self._llm)

    def ask(
        self,
        user_message: str,
        chat_history: List[Dict[str, str]] | None = None,
    ) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
        
        print("---<THINKING>---")

        history = chat_history[:] if chat_history else [m(role="system", content=self._system_prompt, message_type=MessageType.SYSTEM)]
        history.append(m(role="user", content=user_message, message_type=MessageType.USER_FACING))
        
        scope_check = self._scope.check(user_message, history)
        print(f"Checking if the question is in scope: {'In scope' if scope_check.scope == ScopeEnum.IN_SCOPE else 'Out of scope'}")
        print(f"   {scope_check.reasoning}")

        history.append(m(
            role="assistant",
            content=f"🔍 Scope Check: {'In scope' if scope_check.scope == ScopeEnum.IN_SCOPE else 'Out of scope'}",
            reasoning=scope_check.reasoning,
            message_type=MessageType.THINKING
        ))

        if scope_check.scope == ScopeEnum.OUT_OF_SCOPE:
            out_of_scope_response = {
                "content": "I apologize, but I can only answer questions about the Bitext Customer Support Service dataset. Your question appears to be about something else.",
                "reasoning": scope_check.reasoning
            }
            scope_msg = m(
                role="assistant", 
                content=out_of_scope_response["content"], 
                reasoning=out_of_scope_response["reasoning"],
                message_type=MessageType.USER_FACING
            )
            history.append(scope_msg)

            print("---</THINKING>---")
            return out_of_scope_response, history
        
        answer, tool_msgs = self._brain.think(history)
        print("---</THINKING>---\n")
        history.extend(tool_msgs)
        history.append(m(
            role="assistant",
            content=answer["content"],
            reasoning=answer["reasoning"],
            message_type=MessageType.USER_FACING
        ))
        
        return answer, history


    def _discover_capabilities(self):
        func, _ = _TOOL_FUNCS["dataset_info"]
        return func()