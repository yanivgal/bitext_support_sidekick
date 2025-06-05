from pydantic import BaseModel, Field
from message_models.message import MessageType

_system_prompt = (
    "You are thinking out loud before deciding whether to use a tool. "
    "You will be given a conversation history between a user and an assistant. "
    "Your goal is to analyze this conversation and determine what single next action will best move toward fully answering the user's request.\n\n"
    "First, review the conversation history to understand:\n"
    "- What is the user's original request?\n"
    "- What information has already been gathered?\n"
    "- What progress has been made so far?\n\n"
    "Then assess whether the user's original request has already been completely satisfied. "
    "If not, think about what specific piece of information is still missing.\n\n"
    "Finally, decide:\n"
    "- Should you call a tool to get that missing information?\n"
    "- Or do you already have everything needed and should just proceed to respond?\n\n"
    "IMPORTANT GUIDELINES:\n"
    "1. If you have all the information needed, set use_tool=False and provide a clear next_step that summarizes what you will say in your final response.\n"
    "2. If you need multiple pieces of information, prefer to gather them one at a time. This helps maintain clarity and makes it easier to track progress.\n"
    "Respond using the fields:\n"
    "- 'use_tool': true if a tool is needed, false otherwise\n"
    "- 'reasoning': a short explanation of your decision, referencing the conversation history\n"
    "- 'next_step': a one-sentence description of the next action"
)

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
