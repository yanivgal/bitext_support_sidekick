from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any
from brain.graph import agent_graph, AgentState
from chat.message import MessageType, m
from brain.memory_manager import load_user_memory


class Agent:

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        mode: str = "reactive",
    ):
        # For now, keep mode for compatibility but it won't be used in LangGraph version
        if mode not in ["reactive", "plan"]:
            raise ValueError("mode must be 'reactive' or 'plan'")
        
        self._mode = mode
        self._model = model
        # LangGraph graph is already compiled in brain/graph.py

    def ask(
        self,
        user_message: str,
        chat_history: List[Dict[str, str]] | None = None,
        session_id: Optional[str] = None,
    ) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
        
        print("---<THINKING>---")

        # Initialize state
        initial_state = self._initialize_state(user_message, chat_history, session_id)
        
        # Run the graph
        try:
            result = agent_graph.invoke(initial_state, config={"configurable": {"thread_id": session_id or "default"}})
            
            # Extract the final answer and updated messages
            final_answer = result.get("final_answer", "No response generated")
            all_messages = result.get("messages", [])
            
            # Create response in the expected format
            response = m(
                role="assistant",
                content=final_answer,
                message_type=MessageType.USER_FACING
            )
            
            # Return response and full updated history for UI
            print("---</THINKING>---\n")
            return response, all_messages
            
        except Exception as e:
            print(f"Error in LangGraph execution: {e}")
            error_response = m(
                role="assistant",
                content="I encountered an error while processing your request. Please try again.",
                message_type=MessageType.USER_FACING
            )
            return error_response, []

    def _initialize_state(
        self,
        user_message: str,
        chat_history: List[Dict[str, str]] | None = None,
        session_id: Optional[str] = None,
    ) -> AgentState:
        """Initialize the LangGraph state."""
        
        # Get conversation history from LangGraph's checkpointer if session_id is provided
        messages = []
        if session_id:
            stored_messages = self.get_conversation_history(session_id)
            if stored_messages:
                messages = stored_messages[:]
        elif chat_history:
            messages = chat_history[:]
        
        # Add the current user message
        user_msg = m(role="user", content=user_message, message_type=MessageType.USER_FACING)
        messages.append(user_msg)
        
        # Load user memory from file
        user_memory = load_user_memory()
        
        return {
            "messages": messages,
            "user_message": user_message,
            "query_type": None,
            "session_id": session_id,
            "user_memory": user_memory,
            "current_step": "initialized",
            "final_answer": None
        }

    def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Fetch the conversation history for a given session_id from LangGraph's checkpointer.
        Returns the full message history for that session, or an empty list if none exists.
        """
        checkpointer = agent_graph.checkpointer
        
        # For MemorySaver, we need to use the get() method to retrieve state
        try:
            # Try to get the state using the checkpointer's get method
            state = checkpointer.get({"configurable": {"thread_id": session_id}})
            if state:
                # Check if messages are in channel_values
                if "channel_values" in state:
                    channel_values = state["channel_values"]
                    
                    # Look for messages in channel_values
                    if isinstance(channel_values, dict) and "messages" in channel_values:
                        messages = channel_values["messages"]
                        return messages
                
                # Also check if messages are directly in state
                if "messages" in state:
                    messages = state["messages"]
                    return messages
        except Exception as e:
            print(f"Error retrieving conversation history: {e}")
        
        return []

    def convert_langgraph_messages_to_ui_format(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert LangGraph message format back to UI chat_turns format.
        This is used when restoring conversation history for session switching.
        """
        if not messages:
            return []
        
        turns = []
        current_turn = None
        
        for msg in messages:
            if msg.get("message_type") == MessageType.USER_FACING:
                if msg["role"] == "user":
                    # Start a new turn
                    if current_turn:
                        turns.append(current_turn)
                    current_turn = {
                        "user": msg,
                        "thinking": [],
                        "assistant": None,
                        "duration": None
                    }
                elif msg["role"] == "assistant":
                    # Complete the current turn
                    if current_turn:
                        current_turn["assistant"] = msg
            elif current_turn and msg.get("message_type") in [MessageType.THINKING, MessageType.TOOL_CALL, MessageType.TOOL_RESULT]:
                # Add thinking messages to current turn
                current_turn["thinking"].append(msg)
        
        # Add the last turn if it exists
        if current_turn:
            turns.append(current_turn)
        
        return turns