from pydantic import BaseModel, Field
from message_models.message import MessageType

class ReactiveThinkingStep(BaseModel):
    reasoning: str = Field(
        description="A brief explanation of whether a tool call is needed or not, and why."
    )
    use_tool: bool = Field(
        description="True if a tool should be called next, False if no tool is needed."
    )
    next_step: str = Field(
        description="A single clear sentence describing the immediate next action—either call a specific tool or proceed without tools."
    )
    message_type: MessageType = Field(
        default=MessageType.THINKING,
        description="Type of message - always THINKING for this model"
    )
