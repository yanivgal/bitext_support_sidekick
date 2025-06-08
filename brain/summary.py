from typing import Dict, Any
from chat.message import MessageType, m
from chat.service import Service as ChatService
from brain.memory_manager import update_user_memory, load_user_memory

_llm = ChatService("gpt-4o-mini")

_summary_prompt = """
You are a user memory analyzer. Your job is to analyze the current conversation turn and extract key information about the user's preferences, interests, and patterns.

Analyze the user's query and the agent's response to identify:

1. **Interests**: What topics/categories the user is interested in
2. **Query Patterns**: How the user typically asks questions
3. **Preferences**: What kind of responses they prefer (detailed vs concise, structured vs analysis)
4. **Topics Discussed**: What specific topics were covered in this turn

Return a JSON object with the following structure:
{
  "interests": ["topic1", "topic2"],
  "query_patterns": ["pattern1", "pattern2"], 
  "preferences": {
    "data_format": "structured|unstructured|mixed",
    "detail_level": "low|medium|high"
  },
  "topics_discussed": ["topic1", "topic2"],
  "favorite_categories": ["category1", "category2"]
}

Guidelines:
- Only include relevant, non-obvious insights
- Be specific but concise
- Don't repeat information already in memory unless it's reinforced
- Focus on patterns and preferences, not just facts
- If no new insights, return empty arrays/objects
"""

def summary_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the conversation turn and update user memory with new insights.
    This node runs after each agent response to learn about the user.
    """
    user_message = state.get("user_message", "")
    messages = state.get("messages", [])
    query_type = state.get("query_type", "")
    current_user_memory = state.get("user_memory", {})
    
    print(f"🧠 Summary Node: Analyzing conversation turn")
    
    # Skip if this is a memory query itself
    if "remember" in user_message.lower() or "memory" in user_message.lower():
        print(f"   Skipping memory analysis for memory query")
        return state
    
    try:
        # Prepare conversation context for analysis
        conversation_context = []
        for msg in messages[-3:]:  # Last 3 messages for context
            if msg.get("message_type") == MessageType.USER_FACING:
                conversation_context.append(f"{msg['role']}: {msg['content']}")
        
        # Create analysis prompt
        analysis_prompt = f"""
{_summary_prompt}

Current conversation context:
{chr(10).join(conversation_context)}

Query type: {query_type}

Current user memory: {current_user_memory}

Analyze this conversation turn and extract new insights about the user.
"""

        # Get LLM analysis
        response = _llm.chat([
            {"role": "system", "content": analysis_prompt}
        ])
        
        # Parse the response (assuming it returns JSON)
        import json
        try:
            # Try to extract JSON from the response
            content = response.choices[0].message.content
            # Find JSON in the response
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = content[start:end]
                insights = json.loads(json_str)
            else:
                print(f"   No JSON found in response, skipping memory update")
                return state
        except json.JSONDecodeError as e:
            print(f"   Failed to parse JSON from LLM response: {e}")
            return state
        
        # Update user memory with new insights
        if insights:
            print(f"   Extracted insights: {insights}")
            
            # Increment conversation count
            insights["conversation_count"] = current_user_memory.get("conversation_count", 0) + 1
            
            # Update query type preferences
            if query_type in ["structured", "unstructured"]:
                current_prefs = current_user_memory.get("query_types_preferred", {})
                current_prefs[query_type] = current_prefs.get(query_type, 0) + 1
                insights["query_types_preferred"] = current_prefs
            
            # Save to file
            success = update_user_memory(insights)
            if success:
                # Update the state with new memory
                updated_memory = load_user_memory()
                print(f"   Updated user memory successfully")
                return {
                    **state,
                    "user_memory": updated_memory,
                    "current_step": "updated_user_memory"
                }
            else:
                print(f"   Failed to save user memory")
        
    except Exception as e:
        print(f"   Error in summary node: {e}")
    
    return state 