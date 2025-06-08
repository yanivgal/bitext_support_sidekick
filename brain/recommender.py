from typing import Dict, Any, List
from chat.message import MessageType, m
from chat.service import Service as ChatService
from brain.memory_manager import load_user_memory

_llm = ChatService("gpt-4o-mini")

_recommender_prompt = """
You are a next-query recommender for a customer support data analyst chatbot.
Your job is to suggest 2-3 relevant, actionable queries the user might want to ask next, based on:
- The user's summarized memory (interests, preferences, past topics)
- The current conversation history

Guidelines:
- Suggestions must be specific, relevant, and actionable (not generic)
- Use the user's interests, favorite categories, and recent queries to personalize suggestions
- If the user prefers structured data, suggest queries that return lists, counts, or examples
- If the user prefers analysis, suggest summary or pattern-finding queries
- Avoid repeating the last query
- Return a JSON object with a 'suggestions' field (list of strings)
- Do not include explanations, just the suggestions

Example output:
{"suggestions": ["Show me examples from the REFUND category", "Summarize the most common intents", "What are the top 3 issues in the ACCOUNT category?"]}
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
    response = _llm.chat([
        {"role": "system", "content": prompt}
    ])
    
    # Parse suggestions from LLM response
    import json
    suggestions = []
    try:
        content = response.choices[0].message.content
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end != 0:
            json_str = content[start:end]
            parsed = json.loads(json_str)
            suggestions = parsed.get("suggestions", [])
    except Exception as e:
        print(f"❌ Error parsing recommender suggestions: {e}")
    
    # Create RECOMMENDATION message
    rec_msg = m(
        role="assistant",
        content="Here are some queries you might want to try next:",
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