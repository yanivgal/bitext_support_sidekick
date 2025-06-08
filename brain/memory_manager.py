import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

USER_MEMORY_FILE = "user_memory.json"

def load_user_memory() -> Dict[str, Any]:
    """
    Load user memory from the JSON file.
    Returns default memory structure if file doesn't exist.
    """
    default_memory = {
        "user_id": "default_user",
        "interests": [],
        "query_patterns": [],
        "preferences": {
            "data_format": "mixed",
            "detail_level": "medium"
        },
        "conversation_count": 0,
        "last_updated": None,
        "topics_discussed": [],
        "favorite_categories": [],
        "query_types_preferred": {
            "structured": 0,
            "unstructured": 0
        }
    }
    
    try:
        if os.path.exists(USER_MEMORY_FILE):
            with open(USER_MEMORY_FILE, 'r', encoding='utf-8') as f:
                memory = json.load(f)
                print(f"📁 Loaded user memory: {len(memory.get('interests', []))} interests, {memory.get('conversation_count', 0)} conversations")
                return memory
        else:
            print(f"📁 User memory file not found, creating default memory")
            save_user_memory(default_memory)
            return default_memory
    except Exception as e:
        print(f"❌ Error loading user memory: {e}")
        return default_memory

def save_user_memory(memory: Dict[str, Any]) -> bool:
    """
    Save user memory to the JSON file.
    Returns True if successful, False otherwise.
    """
    try:
        # Update timestamp
        memory["last_updated"] = datetime.now().isoformat()
        
        with open(USER_MEMORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(memory, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved user memory: {len(memory.get('interests', []))} interests, {memory.get('conversation_count', 0)} conversations")
        return True
    except Exception as e:
        print(f"❌ Error saving user memory: {e}")
        return False

def update_user_memory(updates: Dict[str, Any]) -> bool:
    """
    Update user memory with new information.
    Merges updates with existing memory and saves to file.
    """
    try:
        memory = load_user_memory()
        
        # Merge updates
        for key, value in updates.items():
            if key in memory:
                if isinstance(memory[key], list) and isinstance(value, list):
                    # Merge lists, avoiding duplicates
                    memory[key] = list(set(memory[key] + value))
                elif isinstance(memory[key], dict) and isinstance(value, dict):
                    # Merge dictionaries
                    memory[key].update(value)
                else:
                    # Replace value
                    memory[key] = value
            else:
                # Add new key
                memory[key] = value
        
        return save_user_memory(memory)
    except Exception as e:
        print(f"❌ Error updating user memory: {e}")
        return False

def get_user_memory_summary() -> str:
    """
    Get a human-readable summary of user memory.
    Used when user asks "What do you remember about me?"
    """
    memory = load_user_memory()
    
    summary_parts = []
    
    # Basic info
    summary_parts.append(f"I've had {memory.get('conversation_count', 0)} conversations with you.")
    
    # Interests
    interests = memory.get('interests', [])
    if interests:
        summary_parts.append(f"You seem interested in: {', '.join(interests)}.")
    
    # Topics discussed
    topics = memory.get('topics_discussed', [])
    if topics:
        summary_parts.append(f"We've discussed: {', '.join(topics)}.")
    
    # Favorite categories
    categories = memory.get('favorite_categories', [])
    if categories:
        summary_parts.append(f"Your favorite categories are: {', '.join(categories)}.")
    
    # Query preferences
    query_prefs = memory.get('query_types_preferred', {})
    structured_count = query_prefs.get('structured', 0)
    unstructured_count = query_prefs.get('unstructured', 0)
    
    if structured_count > unstructured_count:
        summary_parts.append("You prefer structured queries (specific data requests).")
    elif unstructured_count > structured_count:
        summary_parts.append("You prefer analysis and summary queries.")
    else:
        summary_parts.append("You use both structured and unstructured queries equally.")
    
    # Preferences
    prefs = memory.get('preferences', {})
    detail_level = prefs.get('detail_level', 'medium')
    summary_parts.append(f"You prefer {detail_level} level of detail in responses.")
    
    if not summary_parts:
        return "I don't have much information about your preferences yet. Let's chat more!"
    
    return " ".join(summary_parts) 