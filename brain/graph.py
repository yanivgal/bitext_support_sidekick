from typing import Dict, List, Any, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel
from chat.message import MessageType, m
from tools.tools import _TOOL_FUNCS
import json
from brain.classifier import classify_query_node
from brain.structured_agent import structured_agent_node
from brain.unstructured_agent import unstructured_agent_node
from brain.out_of_scope import out_of_scope_node
from brain.summary import summary_node
from brain.memory_manager import load_user_memory, get_user_memory_summary
from brain.recommender import recommender_node


class AgentState(TypedDict):
    """State structure for the LangGraph agent."""
    messages: List[Dict[str, Any]]  # Conversation history
    user_message: str               # Current user input
    query_type: str | None          # Classified query type
    session_id: str | None          # Session identifier
    user_memory: Dict[str, Any]     # User preferences and key info
    current_step: str               # Current processing step
    final_answer: str | None        # Final response to user


def create_agent_graph() -> StateGraph:
    """Create the main agent graph."""
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("classify_query", classify_query_node)
    workflow.add_node("structured_agent", structured_agent_node)
    workflow.add_node("unstructured_agent", unstructured_agent_node)
    workflow.add_node("out_of_scope", out_of_scope_node)
    workflow.add_node("summary", summary_node)
    workflow.add_node("memory_response", memory_response_node)
    workflow.add_node("recommender", recommender_node)
    
    # Set entry point
    workflow.set_entry_point("classify_query")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "classify_query",
        route_query,
        {
            "structured": "structured_agent",
            "unstructured": "unstructured_agent", 
            "out_of_scope": "out_of_scope",
            "memory_query": "memory_response",
            "recommend_query": "recommender"
        }
    )
    
    # Add summary node after each agent response
    workflow.add_edge("structured_agent", "summary")
    workflow.add_edge("unstructured_agent", "summary")
    workflow.add_edge("out_of_scope", "summary")
    
    # Memory response and recommender go directly to END
    workflow.add_edge("memory_response", END)
    workflow.add_edge("recommender", END)
    
    # Summary node goes to END
    workflow.add_edge("summary", END)
    
    return workflow


def route_query(state: AgentState) -> str:
    """Route to appropriate node based on query type."""
    user_message = state.get("user_message", "").lower()
    
    # Check if this is a memory query
    memory_keywords = ["remember", "memory", "what do you know about me", "tell me about myself"]
    if any(keyword in user_message for keyword in memory_keywords):
        return "memory_query"
    
    return state["query_type"]


def memory_response_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle memory queries like "What do you remember about me?"
    """
    user_message = state.get("user_message", "")
    print(f"🧠 Memory Query: Processing '{user_message}'")
    
    # Get memory summary
    memory_summary = get_user_memory_summary()
    
    # Create response
    response = m(
        role="assistant",
        content=memory_summary,
        reasoning="Retrieved user memory summary from stored profile",
        message_type=MessageType.USER_FACING
    )
    
    return {
        **state,
        "messages": state["messages"] + [response],
        "final_answer": response["content"],
        "current_step": "handled_memory_query"
    }


# Create the graph instance
agent_graph = create_agent_graph().compile(checkpointer=MemorySaver()) 