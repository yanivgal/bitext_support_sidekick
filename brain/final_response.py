from pydantic import BaseModel, Field
from chat.message import MessageType

class FinalResponse(BaseModel):
    
    content: str = Field(
        description="The final response content to be shown to the user"
    )
    reasoning: str = Field(
        description="The reasoning behind the final response"
    )
    message_type: MessageType = Field(
        default=MessageType.USER_FACING,
        description="Type of message - always USER_FACING for this model"
    )