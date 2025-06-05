from chat.service import Service as ChatService
from typing import List, Dict
from chat.message import MessageType
from .scope import ScopeCheck

_system_prompt = (
    "You are a scope checker for the Bitext Customer Support Service dataset. "
    "A question is IN SCOPE if it asks about: "
    "- General information about the dataset (e.g., what is the dataset about, what is the purpose of the dataset, etc.)"
    "- General information about the services of the agent (e.g., what services do you offer, what is your purpose, etc.)"
    "- Categories in the dataset (e.g., ACCOUNT, REFUND, ORDER) "
    "- Examples or patterns within categories "
    "- Intent distributions or common patterns "
    "- Any analysis or information that can be derived from the dataset "
    "- Data analysis tasks that use the dataset (e.g., creating FAQs, analyzing patterns, summarizing categories) "
    "- Creating deliverables from the dataset (e.g., reports, summaries, FAQs, guides) "
    "- Queries that require searching or analyzing the dataset using available tools "
    "A question is OUT OF SCOPE if it asks about: "
    "- Public figures or people not in the dataset "
    "- General knowledge not related to customer service "
    "- Topics completely unrelated to the dataset "
    "- Claims about data existence that cannot be verified in the dataset "
    "IMPORTANT:\n"
    "- If the query is vague but likely related to the dataset (e.g., about agents, services, or responses), classify it as 'in_scope'.\n"
    "- If the query is about analyzing the dataset or creating deliverables from it, classify it as 'in_scope'.\n"
    "- If the query requires using any of the dataset's tools or capabilities, classify it as 'in_scope'.\n"
    "- If someone claims something exists in the dataset, you must verify it exists before classifying as 'out_of_scope'.\n"
    "- To verify a claim, use the exact_search tool to check if the claimed text exists in the dataset.\n"
    "- If the exact_search returns no results, classify the claim as 'out_of_scope'.\n"
    "- Only classify as 'in_scope' if the claim can be verified with exact_search.\n"
    "\n"
    "Examples:\n"
    "- 'What services do you offer?' → in_scope\n"
    "- 'Who are you?' → in_scope\n"
    "- 'What categories exist?' → in_scope\n"
    "- 'How do agents typically respond to account-related issues?' → in_scope\n"
    "- 'Create a FAQ about refunds' → in_scope\n"
    "- 'Analyze common patterns in customer questions' → in_scope\n"
    "- 'Search for questions about refunds' → in_scope\n"
    "- 'Tell me about Elon Musk' → out_of_scope\n"
    "- 'What's the weather today?' → out_of_scope\n"
    "- 'I think Benjamin Button appears in the dataset' → in_scope (needs verification)\n"
    "- 'I think Benjamin Button appears in the dataset' → out_of_scope (alreadyverified with exact_search, no matches found)\n"
    "- 'Tell me about Benjamin Button' → out_of_scope (not claiming it's in the dataset)\n"
    "\n"
    "Respond only with: 'in_scope' or 'out_of_scope'."
)

class Checker:
    """
    This class checks if user messages are relevant to the Bitext Customer Support Service dataset.
    Use this to filter out off-topic questions before processing them.
    For example, questions about the weather or celebrities would be considered out of scope,
    while questions about customer service categories or dataset analysis would be in scope.
    """

    def __init__(self, model: str):
        self._model = model
        self._llm = ChatService(model)

    def check(self, user_message: str, chat_history: List[Dict[str, str]] | None = None) -> ScopeCheck:

        # filter chat history to only include user-facing messages
        relevant_context = []
        if chat_history:
            for msg in chat_history:
                if msg['message_type'] == MessageType.USER_FACING:
                    relevant_context.append(msg)
        
        # create context-aware message
        context = ""
        if relevant_context:
            context = "\n".join([
                f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                for msg in relevant_context
            ])
            if context:
                user_message = f"Previous conversation:\n{context}\n\nCurrent message:\n{user_message}"

        response = self._llm.chat(
            messages=[
                {"role": "system", "content": _system_prompt},
                {"role": "user", "content": user_message}
            ],
            response_format=ScopeCheck,
        )
        return response.choices[0].message.parsed