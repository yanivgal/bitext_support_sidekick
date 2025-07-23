from typing import Dict, List, Any, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel
from communication.message_formatter import MessageType, m
from data_analysis.tools import _TOOL_FUNCS
import json
from query_intelligence.query_classifier import classify_query_node
from data_processing.data_query_processor import structured_agent_node
from data_processing.analysis_processor import unstructured_agent_node
from data_processing.out_of_scope_handler import out_of_scope_node
from user_learning.conversation_analyzer import memory_analyzer_node
from user_learning.user_profile_storage import load_user_memory
from user_learning.query_recommender import recommender_node


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
    workflow.add_node("summary", memory_analyzer_node)
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
    return state["query_type"]


def memory_response_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle memory queries like "What do you remember about me?"
    """
    user_message = state.get("user_message", "")
    user_memory = state.get("user_memory", {})
    print(f"🧠 Memory Query: Processing '{user_message}'")
    
    # Track thinking messages for UI display
    thinking_messages = []
    
    # Add initial thinking message
    thinking_msg = m(
        role="assistant",
        content="I need to check what I remember about you from our previous conversations.",
        reasoning="Starting memory query analysis - I need to retrieve and organize the information I've learned about this user",
        message_type=MessageType.THINKING
    )
    thinking_messages.append(thinking_msg)
    print(f"\n{thinking_msg['reasoning']}")
    print(f"My next step should be: {thinking_msg['content']}")
    
    # Add memory retrieval thinking message
    memory_retrieval_msg = m(
        role="assistant",
        content="Retrieving your conversation history, interests, and preferences...",
        reasoning="Accessing stored information about the user's past interactions, interests, and query preferences",
        message_type=MessageType.THINKING
    )
    thinking_messages.append(memory_retrieval_msg)
    print(f"\n{memory_retrieval_msg['reasoning']}")
    print(f"My next step should be: {memory_retrieval_msg['content']}")
    
    # Add memory data result (similar to tool results)
    import json
    memory_result_msg = m(
        role="tool",
        content=json.dumps(user_memory, ensure_ascii=False, indent=2),
        message_type=MessageType.TOOL_RESULT,
        reasoning="Retrieved user memory data from storage",
        tool_call_id="memory_retrieval"
    )
    thinking_messages.append(memory_result_msg)
    print(f"   ✅ Memory retrieval returned {len(user_memory)} key-value pairs")
    
    # Use LLM to generate a natural, human-readable response
    from chat.service import Service as ChatService
    llm = ChatService("gpt-4o-mini")
    
    memory_prompt = f"""
You are a helpful assistant that remembers information about users. The user is asking about what you remember about them.

Here is the raw memory data about this user:
{user_memory}

Please create a natural, conversational response that tells the user what you remember about them. 

Guidelines:
- Be friendly and conversational, not robotic
- If you know their name, use it naturally in the response
- Structure the information clearly with sections if there's a lot of data
- Use bullet points or formatting to make it easy to read
- Focus on the most interesting/relevant information
- If there's not much data, be encouraging about learning more
- Keep it concise but comprehensive
- Use a warm, helpful tone
- For personal info, be respectful and only share what they've explicitly shared

The user asked: "{user_message}"
"""
    
    # Add final thinking message
    final_thinking_msg = m(
        role="assistant",
        content="Now I'll organize this information into a clear, friendly response for you.",
        reasoning="I have the memory data. Now I need to present it in a natural, well-structured way that's easy to understand",
        message_type=MessageType.THINKING
    )
    thinking_messages.append(final_thinking_msg)
    print(f"\n{final_thinking_msg['reasoning']}")
    print(f"My next step should be: {final_thinking_msg['content']}")
    
    # Generate response using LLM
    llm_response = llm.chat([
        {"role": "system", "content": memory_prompt}
    ])
    
    # Create response
    response = m(
        role="assistant",
        content=llm_response.choices[0].message.content,
        reasoning="Retrieved and summarized your conversation history, interests, and preferences from our previous interactions",
        message_type=MessageType.USER_FACING
    )
    
    return {
        **state,
        "messages": state["messages"] + thinking_messages + [response],
        "final_answer": response["content"],
        "current_step": "handled_memory_query",
        "thinking_messages": thinking_messages
    }


# Create the graph instance
agent_graph = create_agent_graph().compile(checkpointer=MemorySaver()) 