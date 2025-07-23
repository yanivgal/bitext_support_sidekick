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
  - Set name_confidence based on clarity: "high" for explicit, "medium" for clear context, "low" for uncertain
  - Set name_source: "explicit" for direct statements, "inferred" for context clues
- If no new insights, return empty arrays/objects
"""

def memory_analyzer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the conversation turn and update user memory with new insights.
    This node runs after each agent response to learn about the user.
    """
    user_message = state.get("user_message", "")
    messages = state.get("messages", [])
    query_type = state.get("query_type", "")
    current_user_memory = state.get("user_memory", {})
    
    print(f"🧠 Summary Node: Analyzing conversation turn")
    
    # Track thinking messages for UI display
    thinking_messages = []
    
    # Skip if this is a memory query itself
    if "remember" in user_message.lower() or "memory" in user_message.lower():
        print(f"   Skipping memory analysis for memory query")
        return state
    
    try:
        # Add initial thinking message with LLM-generated reasoning
        initial_reasoning_prompt = f"""
You are analyzing a conversation to learn about a user. Generate a brief, natural reasoning for why you're starting this analysis.

Context: User asked: "{user_message}"
Query type: {query_type}

Generate a single sentence explaining why you're analyzing this conversation to learn about the user.
"""
        
        llm = _get_llm()
        initial_reasoning_response = llm.chat([
            {"role": "system", "content": initial_reasoning_prompt}
        ])
        initial_reasoning = initial_reasoning_response.choices[0].message.content
        
        thinking_msg = m(
            role="assistant",
            content="I'm analyzing our conversation to learn more about your preferences and interests.",
            reasoning=initial_reasoning,
            message_type=MessageType.THINKING
        )
        thinking_messages.append(thinking_msg)
        print(f"\n{thinking_msg['reasoning']}")
        print(f"My next step should be: {thinking_msg['content']}")
        
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
            # Add error thinking message with LLM-generated reasoning
            error_reasoning_prompt = f"""
You encountered an error while analyzing a conversation. Generate a brief, natural reasoning for this situation.

Error: {e}
User's question: "{user_message}"

Generate a single sentence explaining what happened and that you'll keep trying to learn.
"""
            
            error_reasoning_response = llm.chat([
                {"role": "system", "content": error_reasoning_prompt}
            ])
            error_reasoning = error_reasoning_response.choices[0].message.content
            
            error_msg = m(
                role="assistant",
                content="I had trouble analyzing our conversation this time, but I'll keep trying to learn about your preferences.",
                reasoning=error_reasoning,
                message_type=MessageType.THINKING
            )
            thinking_messages.append(error_msg)
            return {
                **state,
                "messages": state["messages"] + thinking_messages,
                "thinking_messages": thinking_messages
            }
        
        # Update user memory with new insights
        if insights:
            print(f"   Extracted insights: {insights}")
            
            # Add insights analysis thinking message with LLM-generated reasoning
            insights_reasoning_prompt = f"""
You have analyzed a conversation and found insights about a user. Generate a brief, natural reasoning for what you discovered.

Insights found: {insights}
User's question: "{user_message}"

Generate a single sentence explaining what insights you found about the user's preferences and patterns.
"""
            
            insights_reasoning_response = llm.chat([
                {"role": "system", "content": insights_reasoning_prompt}
            ])
            insights_reasoning = insights_reasoning_response.choices[0].message.content
            
            insights_msg = m(
                role="assistant",
                content=f"I found some interesting insights about your preferences and patterns.",
                reasoning=insights_reasoning,
                message_type=MessageType.THINKING
            )
            thinking_messages.append(insights_msg)
            print(f"\n{insights_msg['reasoning']}")
            print(f"My next step should be: {insights_msg['content']}")
            
            # Increment conversation count
            insights["conversation_count"] = current_user_memory.get("conversation_count", 0) + 1
            
            # Update query type preferences
            if query_type in ["structured", "unstructured"]:
                current_prefs = current_user_memory.get("query_types_preferred", {})
                current_prefs[query_type] = current_prefs.get(query_type, 0) + 1
                insights["query_types_preferred"] = current_prefs
            
            # Handle personal info updates with confidence logic
            personal_info_updated = False
            current_personal = current_user_memory.get("personal_info", {})
            
            # Start with current personal info to preserve all existing fields
            updated_personal_info = current_personal.copy()
            
            # Update session tracking automatically (always happens)
            from datetime import datetime
            current_date = datetime.now().isoformat()
            updated_personal_info["last_session_date"] = current_date
            
            # Increment session count
            current_session_count = current_personal.get("session_count", 0)
            updated_personal_info["session_count"] = current_session_count + 1
            
            # Set first session date if not already set
            if not current_personal.get("first_session_date"):
                updated_personal_info["first_session_date"] = current_date
                print(f"   🎉 First session recorded: {current_date}")
            
            print(f"   📅 Session tracking updated: count={updated_personal_info['session_count']}, last={current_date}")
            
            if "personal_info" in insights and insights["personal_info"]:
                new_personal_info = insights["personal_info"]
                
                # Only update name if we have higher confidence or explicit source
                if new_personal_info.get("name") and new_personal_info.get("name") != current_personal.get("name"):
                    new_confidence = new_personal_info.get("name_confidence", "low")
                    current_confidence = current_personal.get("name_confidence", "low")
                    
                    # Confidence hierarchy: high > medium > low
                    confidence_levels = {"low": 1, "medium": 2, "high": 3}
                    
                    if (confidence_levels.get(new_confidence, 0) > confidence_levels.get(current_confidence, 0) or 
                        new_personal_info.get("name_source") == "explicit"):
                        print(f"   🎯 Updating name to '{new_personal_info['name']}' (confidence: {new_confidence})")
                        
                        # Update name-related fields
                        updated_personal_info["name"] = new_personal_info["name"]
                        updated_personal_info["name_confidence"] = new_confidence
                        updated_personal_info["name_source"] = new_personal_info.get("name_source", "unknown")
                        
                        # Add name learning thinking message
                        name_reasoning_prompt = f"""
You have learned a user's name. Generate a brief, natural reasoning for this learning moment.

Name learned: {new_personal_info['name']}
Confidence: {new_confidence}
Source: {new_personal_info.get('name_source', 'unknown')}

Generate a single sentence explaining why learning this name is important for future conversations.
"""
                        
                        name_reasoning_response = llm.chat([
                            {"role": "system", "content": name_reasoning_prompt}
                        ])
                        name_reasoning = name_reasoning_response.choices[0].message.content
                        
                        name_msg = m(
                            role="assistant",
                            content=f"Great! I learned that your name is {new_personal_info['name']}. I'll remember that for our future conversations.",
                            reasoning=name_reasoning,
                            message_type=MessageType.THINKING
                        )
                        thinking_messages.append(name_msg)
                        personal_info_updated = True
                    else:
                        print(f"   ⚠️  Skipping name update - current confidence higher")
                
                # Update session-related fields if they're provided
                if new_personal_info.get("last_session_date"):
                    updated_personal_info["last_session_date"] = new_personal_info["last_session_date"]
                if new_personal_info.get("session_count") is not None:
                    updated_personal_info["session_count"] = new_personal_info["session_count"]
                if new_personal_info.get("first_session_date"):
                    updated_personal_info["first_session_date"] = new_personal_info["first_session_date"]
                
                # Use the updated personal info that preserves all fields
                insights["personal_info"] = updated_personal_info
            
            # Add memory update thinking message
            update_reasoning_prompt = f"""
You are about to save insights about a user to memory. Generate a brief, natural reasoning for this action.

Insights to save: {insights}
User's question: "{user_message}"

Generate a single sentence explaining why saving these insights to memory is important for future conversations.
"""
            
            update_reasoning_response = llm.chat([
                {"role": "system", "content": update_reasoning_prompt}
            ])
            update_reasoning = update_reasoning_response.choices[0].message.content
            
            update_msg = m(
                role="assistant",
                content="I'm updating my memory with what I learned about you from this conversation.",
                reasoning=update_reasoning,
                message_type=MessageType.THINKING
            )
            thinking_messages.append(update_msg)
            print(f"\n{update_msg['reasoning']}")
            print(f"My next step should be: {update_msg['content']}")
            
            # Save to file
            from brain.memory_manager import update_user_memory, load_user_memory
            success = update_user_memory(insights)
            if success:
                # Update the state with new memory
                updated_memory = load_user_memory()
                print(f"   Updated user memory successfully")
                
                # Add completion thinking message
                completion_reasoning_prompt = f"""
You have successfully saved user insights to memory. Generate a brief, natural reasoning for this completion.

Insights saved: {insights}
User's question: "{user_message}"

Generate a single sentence explaining what you accomplished and how it will help future conversations.
"""
                
                completion_reasoning_response = llm.chat([
                    {"role": "system", "content": completion_reasoning_prompt}
                ])
                completion_reasoning = completion_reasoning_response.choices[0].message.content
                
                completion_msg = m(
                    role="assistant",
                    content="Perfect! I've updated my memory with what I learned about you.",
                    reasoning=completion_reasoning,
                    message_type=MessageType.THINKING
                )
                thinking_messages.append(completion_msg)
                
                return {
                    **state,
                    "user_memory": updated_memory,
                    "current_step": "updated_user_memory",
                    "messages": state["messages"] + thinking_messages,
                    "thinking_messages": thinking_messages
                }
            else:
                print(f"   Failed to save user memory")
                return {
                    **state,
                    "messages": state["messages"] + thinking_messages,
                    "thinking_messages": thinking_messages
                }
        else:
            # No insights found, but still show thinking process and save session tracking
            no_insights_reasoning_prompt = f"""
You analyzed a conversation but didn't find new insights about the user. Generate a brief, natural reasoning for this situation.

User's question: "{user_message}"
Query type: {query_type}

Generate a single sentence explaining that you didn't find new insights but are still paying attention.
"""
            
            no_insights_reasoning_response = llm.chat([
                {"role": "system", "content": no_insights_reasoning_prompt}
            ])
            no_insights_reasoning = no_insights_reasoning_response.choices[0].message.content
            
            no_insights_msg = m(
                role="assistant",
                content="I didn't find any new insights to learn from this conversation, but I'm always paying attention to your preferences.",
                reasoning=no_insights_reasoning,
                message_type=MessageType.THINKING
            )
            thinking_messages.append(no_insights_msg)
            
            # Still save session tracking even when no insights are found
            session_update = {
                "personal_info": updated_personal_info,
                "conversation_count": current_user_memory.get("conversation_count", 0) + 1
            }
            
            # Update query type preferences if applicable
            if query_type in ["structured", "unstructured"]:
                current_prefs = current_user_memory.get("query_types_preferred", {})
                current_prefs[query_type] = current_prefs.get(query_type, 0) + 1
                session_update["query_types_preferred"] = current_prefs
            
            # Save session tracking
            from brain.memory_manager import update_user_memory, load_user_memory
            success = update_user_memory(session_update)
            if success:
                print(f"   📅 Session tracking saved successfully")
                updated_memory = load_user_memory()
                return {
                    **state,
                    "user_memory": updated_memory,
                    "current_step": "updated_session_tracking",
                    "messages": state["messages"] + thinking_messages,
                    "thinking_messages": thinking_messages
                }
            else:
                print(f"   ⚠️  Failed to save session tracking")
                return {
                    **state,
                    "messages": state["messages"] + thinking_messages,
                    "thinking_messages": thinking_messages
                }
        
    except Exception as e:
        print(f"   Error in summary node: {e}")
        # Add error thinking message with LLM-generated reasoning
        try:
            llm = _get_llm()
            exception_reasoning_prompt = f"""
You encountered an exception while trying to learn from a conversation. Generate a brief, natural reasoning for this situation.

Exception: {e}
User's question: "{user_message}"

Generate a single sentence explaining that you encountered an error but will keep trying.
"""
            
            exception_reasoning_response = llm.chat([
                {"role": "system", "content": exception_reasoning_prompt}
            ])
            exception_reasoning = exception_reasoning_response.choices[0].message.content
        except Exception as llm_error:
            print(f"   ⚠️  Could not generate error reasoning: {llm_error}")
            exception_reasoning = "I encountered an error while trying to learn from our conversation, but I'll keep trying."
        
        error_msg = m(
            role="assistant",
            content="I encountered an error while trying to learn from our conversation, but I'll keep trying.",
            reasoning=exception_reasoning,
            message_type=MessageType.THINKING
        )
        thinking_messages.append(error_msg)
        return {
            **state,
            "messages": state["messages"] + thinking_messages,
            "thinking_messages": thinking_messages
        } 