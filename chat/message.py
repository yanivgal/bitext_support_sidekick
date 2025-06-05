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

def m(
    role: str,
    content: str,
    message_type: MessageType,
    reasoning: str | None = None,
    tool_calls: List[Dict] | None = None,
    tool_call_id: str | None = None
) -> Dict:
    """Create a Message object with the given parameters and convert it to a dictionary."""
    return Message(
        role=role,
        content=content,
        message_type=message_type,
        reasoning=reasoning,
        tool_calls=tool_calls,
        tool_call_id=tool_call_id
    ).model_dump()