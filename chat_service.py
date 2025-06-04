import openai
from typing import List, Dict, Any
from pydantic import BaseModel
from message_models.message import MessageType

class ChatService:
    """Service for interacting with OpenAI chat completions API."""
    
    def __init__(self, model: str):
        self._model = model
        self._client = openai.OpenAI()

    def chat(
        self,
        messages: List[Dict[str, str | Dict[str, Any]]],
        tools_json: List[Dict[str, Any]] | None = None,
        response_format: type[BaseModel] | None = None,
    ):
        # Convert any MessageType enums to strings in the messages
        def convert_message_types(msg: Dict) -> Dict:
            if isinstance(msg, dict):
                return {
                    k: v.value if isinstance(v, MessageType) else v
                    for k, v in msg.items()
                }
            return msg

        messages = [convert_message_types(msg) for msg in messages]

        if response_format:
            return self._client.beta.chat.completions.parse(
                model=self._model,
                messages=messages,
                response_format=response_format
            )

        kwargs = {
            "model": self._model,
            "messages": messages,
        }
        if tools_json:
            kwargs["tools"] = tools_json
            kwargs["tool_choice"] = "auto"

        return self._client.chat.completions.create(**kwargs)