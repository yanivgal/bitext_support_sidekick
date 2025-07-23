from enum import Enum
from typing import Optional, List, Dict
from pydantic import BaseModel

class MessageType(Enum):
    SYSTEM = "system"
    THINKING = "thinking"
    USER_FACING = "user_facing"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"

class Message(BaseModel):
    role: str
    content: str
    message_type: MessageType
    reasoning: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None

    model_config = {
        "json_encoders": {
            MessageType: lambda v: v.value
        }
    }

def m(role, content, message_type, reasoning=None, tool_calls=None, tool_call_id=None, **kwargs):
    msg = {
        "role": role,
        "content": content,
        "message_type": message_type,
    }
    if reasoning is not None:
        msg["reasoning"] = reasoning
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    if tool_call_id is not None:
        msg["tool_call_id"] = tool_call_id
    msg.update(kwargs)  # Allow extra fields like 'suggestions'
    return msg