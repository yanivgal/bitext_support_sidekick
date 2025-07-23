from typing import Dict, Any
from chat.message import MessageType, m

def out_of_scope_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle out-of-scope queries with proper responses and reasoning.
    """
    user_message = state["user_message"]
    print(f"❌ Out-of-Scope Handler: Processing '{user_message}'")
    
    # Get the scope reasoning from the classifier
    scope_reasoning = state.get("scope_reasoning", "Query appears to be outside the scope of the Bitext dataset.")
    
    response_content = (
        "I apologize, but I can only answer questions about the Bitext Customer Support Service dataset. "
        "Your question appears to be about something else. "
        "I can help you with questions about customer service categories, intents, responses, and data analysis."
    )
    
    response = m(
        role="assistant",
        content=response_content,
        reasoning=f"Out-of-scope detection: {scope_reasoning}",
        message_type=MessageType.USER_FACING
    )
    
    # Always append to the existing message history
    return {
        **state,
        "messages": state["messages"] + [response],
        "final_answer": response["content"],
        "current_step": "handled_out_of_scope_with_agent"
    } 