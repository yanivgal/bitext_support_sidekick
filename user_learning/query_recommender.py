from typing import Dict, Any, List
from chat.message import MessageType, m

# Initialize LLM service lazily to avoid import-time API key issues
_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        from chat.service import Service as ChatService
        _llm = ChatService("gpt-4o-mini")
    return _llm

_recommender_prompt = """
You are a final recommendation generator for a customer support data analyst chatbot.
Your job is to provide 2-3 specific, actionable query suggestions after the user has had a conversation about what they want to explore.

IMPORTANT: This is the final step - provide concrete, clickable suggestions based on the conversation that just happened.

Guidelines:
- Focus on 2-3 specific, actionable suggestions (not generic options)
- Make suggestions that directly address what the user expressed interest in during the conversation
- Be very specific: "Show me the top 3 refund issues" not "look at refunds"
- If they mentioned specific categories, focus on those
- If they mentioned specific types of analysis, provide queries for that
- Avoid generic suggestions like "explore categories" or "analyze patterns"
- Return a JSON object with a 'suggestions' field (list of 2-3 strings) and a 'conversation' field (string with your reasoning)

Example output:
{
  "conversation": "Based on your interest in customer service patterns and your recent focus on refund issues, I think you'd find these specific questions valuable for understanding customer pain points...",
  "suggestions": ["Show me the top 3 refund issues by frequency", "What are the most common customer intents in the REFUND category?"]
}
"""

def recommender_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Suggest next queries based on user memory and conversation history.
    Returns a RECOMMENDATION message with suggestions for the UI.
    """
    user_memory = state.get("user_memory", {})
    messages = state.get("messages", [])
    
    # Prepare context for the LLM
    history = []
    for msg in messages[-5:]:  # Last 5 user-facing messages for context
        if msg.get("message_type") == MessageType.USER_FACING:
            history.append(f"{msg['role']}: {msg['content']}")
    
    prompt = f"""
{_recommender_prompt}

User memory:
{user_memory}

Recent conversation:
{chr(10).join(history)}
"""
    
    # Call LLM
    response = _get_llm().chat([
        {"role": "system", "content": prompt}
    ])
    
    # Parse suggestions from LLM response
    import json
    suggestions = []
    conversation = ""
    try:
        content = response.choices[0].message.content
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end != 0:
            json_str = content[start:end]
            parsed = json.loads(json_str)
            suggestions = parsed.get("suggestions", [])
            conversation = parsed.get("conversation", "")
    except Exception as e:
        print(f"❌ Error parsing recommender suggestions: {e}")
    
    # Create RECOMMENDATION message with conversation
    if conversation:
        # First show the conversation
        conversation_msg = m(
            role="assistant",
            content=conversation,
            reasoning="Explaining reasoning for query suggestions based on user's interests and preferences",
            message_type=MessageType.USER_FACING
        )
        
        # Then show the recommendations
        rec_msg = m(
            role="assistant",
            content="Based on our conversation and your interests, here are some specific queries you might find interesting. You can click any of these to explore further:",
            message_type="RECOMMENDATION",
            suggestions=suggestions
        )
        
        return {
            **state,
            "messages": state["messages"] + [conversation_msg, rec_msg],
            "final_answer": conversation_msg["content"],
            "current_step": "recommended_queries_with_conversation",
            "recommendations": suggestions
        }
    else:
        # Fallback to old format if no conversation
        rec_msg = m(
            role="assistant",
            content="Based on your interests and the dataset, here are some queries you might want to explore. Click any of these to get started:",
            message_type="RECOMMENDATION",
            suggestions=suggestions
        )
    
    return {
        **state,
        "messages": state["messages"] + [rec_msg],
        "final_answer": rec_msg["content"],
        "current_step": "recommended_queries",
        "recommendations": suggestions
    } 