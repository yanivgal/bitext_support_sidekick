from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import pandas as pd
import openai
from tools.tools import _TOOL_FUNCS, TOOLS_SCHEMA
from message_models.scope import ScopeCheck, ScopeEnum
from message_models.reactive_thinking_step import ReactiveThinkingStep
from message_models.final_response import FinalResponse
from message_models.message import Message, MessageType, m
from pydantic import BaseModel
from message_models.planning_thinking import PlanningThinking, PlanningStep
from system_prompts import get_system_prompt

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
        self._client = openai.OpenAI()
        self._capabilities = self._discover_capabilities()
        self._system_prompt = get_system_prompt(self._mode, self._capabilities)

    def ask(
        self,
        user_message: str,
        chat_history: List[Dict[str, str]] | None = None,
    ) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
        
        print("---<THINKING>---")

        history = chat_history[:] if chat_history else [m(role="system", content=self._system_prompt, message_type=MessageType.SYSTEM)]
        history.append(m(role="user", content=user_message, message_type=MessageType.USER_FACING))
        
        scope_check = self._check_scope(user_message, history)
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
        
        if self._mode == "reactive":
            answer, tool_msgs = self._reactive_cycle(history)
            history.extend(tool_msgs)
            history.append(m(
                role="assistant",
                content=answer["content"],
                reasoning=answer["reasoning"],
                message_type=MessageType.USER_FACING
            ))
            return answer, history

        if self._mode == "plan":
            answer, intermediate = self._plan_then_execute(history)
            history.extend(intermediate)
            history.append(m(
                role="assistant",
                content=answer["content"],
                reasoning=answer["reasoning"],
                message_type=MessageType.USER_FACING
            ))
            return answer, history

        raise ValueError("mode must be 'reactive' or 'plan'")


    def _chat(
        self,
        messages: List[Dict[str, str | Dict[str, Any]]],
        tools_json: List[Dict[str, Any]] | None = None,
        response_format: type[BaseModel] | None = None,
        tool_choice: Dict[str, Any] | str | None = None,
    ):
        # Convert any MessageType enums to strings in the messages
        def convert_message_types(msg: Dict) -> Dict:
            if isinstance(msg, dict):
                return {
                    k: v.value if isinstance(v, MessageType) else v
                    for k, v in msg.items()
                }
            return msg

        messages = [convert_message_types(msg) for msg in messages]

        if response_format:
            return self._client.beta.chat.completions.parse(
                model=self._model,
                messages=messages,
                response_format=response_format
            )

        kwargs = {
            "model": self._model,
            "messages": messages,
        }
        if tools_json:
            kwargs["tools"] = tools_json
            kwargs["tool_choice"] = "auto"

        return self._client.chat.completions.create(**kwargs)




    def _get_final_response(self, messages: List[Dict[str, str]]) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
        
        resp = self._chat(
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

    def _reactive_cycle(self, messages):
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
                answer, _ = self._get_final_response(working)
                print("---</THINKING>---\n")
                return answer, new_msgs

            # Now get the tool calls - request only one tool
            resp = self._chat(
                working, 
                tools_json=TOOLS_SCHEMA
            )
            msg = resp.choices[0].message

            if msg.tool_calls:
                working, new_msgs = self._handle_tool_calls(msg, working, new_msgs)
                continue

            # no tool calls → final answer
            answer, _ = self._get_final_response(working)
            print("---</THINKING>---\n")
            return answer, new_msgs

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
            result = self._execute_tool(name, args)

            tool_msg = m(
                role="tool",
                content=json.dumps(result, ensure_ascii=False),
                message_type=MessageType.TOOL_RESULT,
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


    def _plan_then_execute(self, messages):
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
                resp = self._chat(working, tools_json=TOOLS_SCHEMA)
                msg = resp.choices[0].message
                
                if msg.tool_calls:
                    working, new_msgs = self._handle_tool_calls(msg, working, new_msgs)
        
        answer, _ = self._get_final_response(working)
        print("---</THINKING>---\n")
        
        return {
            "content": answer["content"],
            "reasoning": answer["reasoning"]
        }, new_msgs


    def _discover_capabilities(self):
        
        # categories = self._execute_tool("get_categories", {})
        dataset_info = self._execute_tool("dataset_info", {})
        
        return dataset_info


    def _check_scope(self, user_message: str, chat_history: List[Dict[str, str]] | None = None) -> ScopeCheck:

        # filter chat history to only include user-facing messages
        relevant_context = []
        if chat_history:
            for msg in chat_history:
                if msg['message_type'] == MessageType.USER_FACING:
                    relevant_context.append(msg)
        
        # create context-aware message
        context = ""
        if relevant_context:
            context = "\n".join([
                f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                for msg in relevant_context
            ])
            if context:
                user_message = f"Previous conversation:\n{context}\n\nCurrent message:\n{user_message}"

        response = self._chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a scope checker for the Bitext Customer Support Service dataset. "
                        "A question is IN SCOPE if it asks about: "
                        "- General information about the dataset (e.g., what is the dataset about, what is the purpose of the dataset, etc.)"
                        "- General information about the services of the agent (e.g., what services do you offer, what is your purpose, etc.)"
                        "- Categories in the dataset (e.g., ACCOUNT, REFUND, ORDER) "
                        "- Examples or patterns within categories "
                        "- Intent distributions or common patterns "
                        "- Any analysis or information that can be derived from the dataset "
                        "- Data analysis tasks that use the dataset (e.g., creating FAQs, analyzing patterns, summarizing categories) "
                        "- Creating deliverables from the dataset (e.g., reports, summaries, FAQs, guides) "
                        "- Queries that require searching or analyzing the dataset using available tools "
                        "A question is OUT OF SCOPE if it asks about: "
                        "- Public figures or people not in the dataset "
                        "- General knowledge not related to customer service "
                        "- Topics completely unrelated to the dataset "
                        "- Claims about data existence that cannot be verified in the dataset "
                        "IMPORTANT:\n"
                        "- If the query is vague but likely related to the dataset (e.g., about agents, services, or responses), classify it as 'in_scope'.\n"
                        "- If the query is about analyzing the dataset or creating deliverables from it, classify it as 'in_scope'.\n"
                        "- If the query requires using any of the dataset's tools or capabilities, classify it as 'in_scope'.\n"
                        "- If someone claims something exists in the dataset, you must verify it exists before classifying as 'out_of_scope'.\n"
                        "- To verify a claim, use the exact_search tool to check if the claimed text exists in the dataset.\n"
                        "- If the exact_search returns no results, classify the claim as 'out_of_scope'.\n"
                        "- Only classify as 'in_scope' if the claim can be verified with exact_search.\n"
                        "\n"
                        "Examples:\n"
                        "- 'What services do you offer?' → in_scope\n"
                        "- 'Who are you?' → in_scope\n"
                        "- 'What categories exist?' → in_scope\n"
                        "- 'How do agents typically respond to account-related issues?' → in_scope\n"
                        "- 'Create a FAQ about refunds' → in_scope\n"
                        "- 'Analyze common patterns in customer questions' → in_scope\n"
                        "- 'Search for questions about refunds' → in_scope\n"
                        "- 'Tell me about Elon Musk' → out_of_scope\n"
                        "- 'What's the weather today?' → out_of_scope\n"
                        "- 'I think Benjamin Button appears in the dataset' → in_scope (needs verification)\n"
                        "- 'I think Benjamin Button appears in the dataset' → out_of_scope (alreadyverified with exact_search, no matches found)\n"
                        "- 'Tell me about Benjamin Button' → out_of_scope (not claiming it's in the dataset)\n"
                        "\n"
                        "Respond only with: 'in_scope' or 'out_of_scope'."
                    )
                },
                {"role": "user", "content": user_message}
            ],
            response_format=ScopeCheck,
        )
        return response.choices[0].message.parsed
    

    def _think_next_step(self, messages: list[dict]) -> ReactiveThinkingStep:

        # Convert the first system message to user message if it exists
        modified_messages = messages.copy()
        if modified_messages and modified_messages[0]["role"] == "system":
            modified_messages[0] = {
                "role": "user",
                "content": modified_messages[0]["content"]
            }

        # Add new system message at the beginning
        modified_messages.insert(0, {
            "role": "system",
            "content": (
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
        })

        response = self._chat(
            modified_messages,
            tools_json=None,
            response_format=ReactiveThinkingStep,
        )
        return response.choices[0].message.parsed
    
    def _plan_thinking(self, messages: list[dict]) -> PlanningThinking:
        
        response = self._chat(
            messages + [{
                "role": "system",
                "content": (
                    "Create a structured plan to answer the user's question. "
                    "Break down the work into clear steps, considering dependencies between steps. "
                    "For each step, specify what needs to be done and what we expect to get from it."
                )
            }],
            tools_json=None,
            response_format=PlanningThinking
        )
        return response.choices[0].message.parsed




    

    def _normalize_category(self, category: str) -> str:
        """Normalize category name to match dataset format."""
        if not category:
            return category
        return category.upper()

    def _execute_tool(self, name: str, args: Dict[str, Any]):
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
    
