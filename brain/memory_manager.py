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

 