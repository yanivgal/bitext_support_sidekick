from typing import Dict, Any
from chat.message import MessageType, m

# Initialize LLM service lazily to avoid import-time API key issues
_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        from chat.service import Service as ChatService
        _llm = ChatService("gpt-4o-mini")
    return _llm

_summary_prompt = """
You are a user memory analyzer. Your job is to analyze the current conversation turn and extract key information about the user's preferences, interests, patterns, and personal identity.

Analyze the user's query and the agent's response to identify:

1. **Interests**: What topics/categories the user is interested in
2. **Query Patterns**: How the user typically asks questions
3. **Preferences**: What kind of responses they prefer (detailed vs concise, structured vs analysis)
4. **Topics Discussed**: What specific topics were covered in this turn
5. **Personal Identity**: Name, greeting preferences, and personal information

Return a JSON object with the following structure:
{
  "interests": ["topic1", "topic2"],
  "query_patterns": ["pattern1", "pattern2"], 
  "preferences": {
    "data_format": "structured|unstructured|mixed",
    "detail_level": "low|medium|high"
  },
  "topics_discussed": ["topic1", "topic2"],
  "favorite_categories": ["category1", "category2"],
  "personal_info": {
    "name": "string or null",
    "preferred_greeting": "string or null",
    "name_confidence": "low|medium|high",
    "name_source": "explicit|inferred|null"
  }
}

Guidelines:
- Only include relevant, non-obvious insights
- Be specific but concise
- Don't repeat information already in memory unless it's reinforced
- Focus on patterns and preferences, not just facts
- For personal identity:
  - Look for explicit name introductions: "My name is...", "I'm...", "Call me..."
  - Look for greeting patterns: "Hi", "Hello", "Hey", "Good morning", etc.
  - Set name_confidence based on clarity: "high" for explicit, "medium" for clear context, "low" for uncertain
  - Set name_source: "explicit" for direct statements, "inferred" for context clues
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
        llm = _get_llm()
        response = llm.chat([
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
            
            # Handle personal info updates with confidence logic
            if "personal_info" in insights and insights["personal_info"]:
                personal_info = insights["personal_info"]
                current_personal = current_user_memory.get("personal_info", {})
                
                # Only update name if we have higher confidence or explicit source
                if personal_info.get("name") and personal_info.get("name") != current_personal.get("name"):
                    new_confidence = personal_info.get("name_confidence", "low")
                    current_confidence = current_personal.get("name_confidence", "low")
                    
                    # Confidence hierarchy: high > medium > low
                    confidence_levels = {"low": 1, "medium": 2, "high": 3}
                    
                    if (confidence_levels.get(new_confidence, 0) > confidence_levels.get(current_confidence, 0) or 
                        personal_info.get("name_source") == "explicit"):
                        print(f"   🎯 Updating name to '{personal_info['name']}' (confidence: {new_confidence})")
                    else:
                        print(f"   ⚠️  Skipping name update - current confidence higher")
                        personal_info["name"] = current_personal.get("name")
                        personal_info["name_confidence"] = current_personal.get("name_confidence")
                        personal_info["name_source"] = current_personal.get("name_source")
                
                # Update greeting preference if detected
                if personal_info.get("preferred_greeting") and not current_personal.get("preferred_greeting"):
                    print(f"   👋 Detected greeting preference: '{personal_info['preferred_greeting']}'")
                
                insights["personal_info"] = personal_info
            
            # Save to file
            from brain.memory_manager import update_user_memory, load_user_memory
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