from typing import Dict, Any
from pydantic import BaseModel, Field
from communication.llm_communicator import Service as ChatService
from communication.message_formatter import MessageType
from .scope_checker import Checker

# LLM-based classifier system prompt for structured/unstructured
_classifier_prompt = (
    "You are a query classifier for the Bitext Customer Support Service dataset.\n"
    "Classify the user's query as one of the following types:\n"
    "- structured: Direct, specific requests for data (e.g., counts, lists, examples, exact matches, statistics).\n"
    "- unstructured: Requests for summaries, analysis, explanations, or pattern discovery (e.g., summarize, analyze, explain, find common patterns, generate FAQ).\n"
    "- recommend_query: Requests for advice, suggestions, or recommendations about what to ask next.\n"
    "- memory_query: Requests about user memory, preferences, or what the agent remembers about the user.\n"
    "\n"
    "Instructions:\n"
    "- If the query is about general dataset info, categories, intents, or direct data retrieval, classify as 'structured'.\n"
    "- If the query is about summarizing, analyzing, or discovering patterns, classify as 'unstructured'.\n"
    "- If the query is about what to ask next, or asks for advice, suggestions, or recommendations, classify as 'recommend_query'.\n"
    "- If the query is about user memory, preferences, or what the agent remembers about the user, classify as 'memory_query'.\n"
    "\n"
    "IMPORTANT DISTINCTION:\n"
    "- 'What can you do?' or 'What are your capabilities?' → unstructured (conversation about capabilities)\n"
    "- 'What do you think I should ask next?' or 'Why do you think that?' → unstructured (conversation about suggestions)\n"
    "- 'What should I ask next?' or 'What should I ask now?' or 'Suggest a good next question' → unstructured (conversation about suggestions)\n"
    "- 'Advise me what to query next' or 'What would you recommend?' → unstructured (conversation about suggestions)\n"
    "- 'Give me recommendations' or 'Show me recommendations' or 'Give me buttons' → recommend_query (direct recommendations with buttons)\n"
    "\n"
    "Examples:\n"
    "- 'What are the most frequent categories?' → structured\n"
    "- 'Show examples of Category X' → structured\n"
    "- 'Summarize the ACCOUNT category' → unstructured\n"
    "- 'Analyze common patterns in customer questions' → unstructured\n"
    "- 'What can you do?' → unstructured\n"
    "- 'What are your capabilities?' → unstructured\n"
    "- 'What do you think I should ask next?' → unstructured\n"
    "- 'Why do you think that?' → unstructured\n"
    "- 'What should I ask next?' → unstructured\n"
    "- 'What should I ask now?' → unstructured\n"
    "- 'Suggest a good next question' → unstructured\n"
    "- 'Advise me what to query next' → unstructured\n"
    "- 'What would you recommend?' → unstructured\n"
    "- 'Give me recommendations' → recommend_query\n"
    "- 'Show me recommendations' → recommend_query\n"
    "- 'Give me buttons' → recommend_query\n"
    "- 'What do you remember about me?' → memory_query\n"
    "- 'Tell me about my preferences' → memory_query\n"
    "- 'What do you know about me?' → memory_query\n"
    "\n"
    "Respond only with a JSON object with two fields: 'query_type' (structured, unstructured, recommend_query, or memory_query) and 'reasoning' (a short explanation for your classification)."
)

class QueryClassification(BaseModel):
    query_type: str = Field(..., description="structured, unstructured, recommend_query, or memory_query")
    reasoning: str = Field(..., description="Short explanation for the classification")

# Initialize these lazily to avoid import-time API key issues
_llm = None
_scope_checker = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatService("gpt-4o-mini")
    return _llm

def _get_scope_checker():
    global _scope_checker
    if _scope_checker is None:
        _scope_checker = Checker(model="gpt-4o-mini")
    return _scope_checker

def classify_query_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    First use the old scope checker. If in-scope, use the LLM to classify as structured/unstructured/recommend_query. If out-of-scope, return immediately.
    """
    user_message = state["user_message"]
    chat_history = state.get("messages", [])

    print(f"🔍 Scope checking query: {user_message}")
    scope_checker = _get_scope_checker()
    scope_result = scope_checker.check(user_message, chat_history)
    print(f"   Scope check result: {scope_result.scope.value}")
    print(f"   Scope reasoning: {scope_result.reasoning}")

    if scope_result.scope.value.lower() == "out of scope":
        print("   → Classified as OUT OF SCOPE by scope checker")
        print(f"   → Reasoning: {scope_result.reasoning}")
        return {
            **state,
            "query_type": "out_of_scope",
            "current_step": "classified_query_out_of_scope",
            "scope_reasoning": scope_result.reasoning
        }

    # If in-scope, use LLM to classify as structured/unstructured/recommend_query
    print(f"🔍 LLM Classifying in-scope query: {user_message}")
    llm = _get_llm()
    messages = [
        {"role": "system", "content": _classifier_prompt},
        {"role": "user", "content": user_message}
    ]
    response = llm.chat(messages, response_format=QueryClassification)
    classification = response.choices[0].message.parsed
    print(f"   → Classified as {classification.query_type.upper()}: {classification.reasoning}")

    return {
        **state,
        "query_type": classification.query_type,
        "current_step": f"classified_query_{classification.query_type}",
        "scope_reasoning": scope_result.reasoning,
        "classification_reasoning": classification.reasoning
    } 