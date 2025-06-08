import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from agent import Agent
from chat.message import MessageType, Message
import time
import uuid
from brain.graph import agent_graph

from dotenv import load_dotenv

load_dotenv()

# --- Session Management Utilities ---
SESSION_LIST_KEY = "session_ids"
CURRENT_SESSION_KEY = "current_session_id"


def get_session_ids():
    if SESSION_LIST_KEY not in st.session_state:
        st.session_state[SESSION_LIST_KEY] = []
    return st.session_state[SESSION_LIST_KEY]

def add_session_id(session_id: str):
    session_ids = get_session_ids()
    if session_id not in session_ids:
        session_ids.append(session_id)
        st.session_state[SESSION_LIST_KEY] = session_ids

def set_current_session(session_id: str):
    st.session_state[CURRENT_SESSION_KEY] = session_id
    add_session_id(session_id)

def get_current_session():
    if CURRENT_SESSION_KEY not in st.session_state:
        # Generate a default session ID if none exists
        default_id = f"session_{uuid.uuid4().hex[:8]}"
        set_current_session(default_id)
    return st.session_state[CURRENT_SESSION_KEY]

# Helper to convert flat message history to chat_turns format
def restore_chat_turns_from_history(history_msgs):
    chat_turns = []
    i = 0
    while i < len(history_msgs):
        msg = history_msgs[i]
        if msg["role"] == "user":
            turn = {"user": msg, "thinking": [], "assistant": None, "duration": None}
            # Collect thinking and assistant messages after user
            j = i + 1
            while j < len(history_msgs) and history_msgs[j]["role"] != "user":
                if history_msgs[j]["message_type"] == MessageType.USER_FACING:
                    turn["assistant"] = history_msgs[j]
                else:
                    turn["thinking"].append(history_msgs[j])
                j += 1
            chat_turns.append(turn)
            i = j
        else:
            i += 1
    return chat_turns

# --- UI Components ---
def sidebar_session_controls():
    st.sidebar.subheader("Conversations")
    session_ids = get_session_ids()
    current_session = get_current_session()

    # Dropdown for existing sessions
    if session_ids:
        selected = st.sidebar.selectbox(
            "Select a conversation:",
            session_ids,
            index=session_ids.index(current_session) if current_session in session_ids else 0,
            key="session_select_box"
        )
        if selected != current_session:
            set_current_session(selected)
            # Restore conversation history from LangGraph memory
            agent = st.session_state.get('agent')
            if agent:
                stored_messages = agent.get_conversation_history(selected)
                restored_turns = agent.convert_langgraph_messages_to_ui_format(stored_messages)
                st.session_state.chat_turns = restored_turns
            else:
                st.session_state.chat_turns = []
            st.rerun()

    # Button to start a new conversation
    if st.sidebar.button("New Conversation"):
        new_id = f"session_{uuid.uuid4().hex[:8]}"
        set_current_session(new_id)
        st.session_state.chat_turns = []
        st.rerun()

    # Text input for custom session ID
    custom_id = st.sidebar.text_input("Or enter a session ID:", value="", key="custom_session_id")
    if custom_id:
        if st.sidebar.button("Switch to Custom Session"):
            set_current_session(custom_id)
            # Restore conversation history from LangGraph memory
            agent = st.session_state.get('agent')
            if agent:
                stored_messages = agent.get_conversation_history(custom_id)
                restored_turns = agent.convert_langgraph_messages_to_ui_format(stored_messages)
                st.session_state.chat_turns = restored_turns
            else:
                st.session_state.chat_turns = []
            st.rerun()

    st.sidebar.caption(f"Current session: {get_current_session()}")

# --- Chat UI Logic ---
def _format_duration(seconds: float) -> str:
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    parts = []
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if seconds or not parts:
        parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")
    return " and ".join(parts)

def display_thinking_messages(messages, duration: float | None = None):
    if not messages:
        return
    if duration is not None:
        time_str = _format_duration(duration)
        label = f"🤔 Thought for {time_str}"
    else:
        label = "Thinking...  🤔"
    with st.expander(label):
        tool_calls = []
        for i, msg in enumerate(messages):
            if msg["message_type"] == MessageType.TOOL_CALL and "tool_calls" in msg:
                for tool_call in msg["tool_calls"]:
                    tool_calls.append((i, tool_call))
        current_tool_call_idx = 0
        for i, msg in enumerate(messages):
            if msg["message_type"] == MessageType.THINKING:
                st.write("🤔  " + msg["content"])
                if "reasoning" in msg:
                    st.markdown(f"> **Reasoning:** {msg['reasoning']}")
            elif msg["message_type"] == MessageType.TOOL_CALL:
                if tool_calls and current_tool_call_idx == 0:
                    st.write(f"🔧  {msg['content']}")
            elif msg["message_type"] == MessageType.TOOL_RESULT:
                if current_tool_call_idx < len(tool_calls):
                    tool_call_idx, tool_call = tool_calls[current_tool_call_idx]
                    st.markdown(f"> 🛠️  Calling {tool_call['function']['name']} tool with args: {tool_call['function']['arguments']}")
                    current_tool_call_idx += 1
                st.markdown(f"""> ✅  Tool Result:  
                            {msg['content']}
                            """)

def display_message(message):
    if message["role"] == "system":
        return
    if message["message_type"] == MessageType.USER_FACING:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant" and "reasoning" in message:
                with st.expander("Agent's Reasoning"):
                    st.write(message["reasoning"])
    elif message["message_type"] == "RECOMMENDATION":
        with st.chat_message("assistant"):
            st.write(message["content"])
            suggestions = message.get("suggestions", [])
            if suggestions:
                st.markdown("**Suggestions:**")
                for i, suggestion in enumerate(suggestions):
                    if st.button(suggestion, key=f"suggestion_{suggestion}_{i}"):
                        st.session_state.suggested_prompt = suggestion
                        st.rerun()

def main():
    st.title("🤖 Bitext Support Sidekick")
    st.caption("Your friendly neighborhood data detective! I'll help you crack the case of customer conversations, decode intents, and make your support experience less 'support-ive' and more 'awesome-ive'! 🕵️‍♂️")

    # --- Sidebar: Session Management ---
    sidebar_session_controls()
    current_session_id = get_current_session()

    # Add mode toggle in the sidebar
    with st.sidebar:
        st.subheader("Agent Mode")
        mode = st.radio(
            "Select agent mode:",
            ["reactive", "plan"],
            key="agent_mode"
        )
        st.caption("⚠️ Switching modes will reset the conversation")
        st.caption("💡 reactive: step-by-step thinking | plan: creates a plan first")

    # Initialize the agent and chat_turns on mode change or first run
    if 'agent' not in st.session_state or st.session_state.get('current_mode') != mode:
        st.session_state.agent = Agent(mode=mode)
        st.session_state.current_mode = mode
        st.session_state.chat_turns = []
    if 'chat_turns' not in st.session_state:
        st.session_state.chat_turns = []

    # Chat input or suggested prompt
    prompt = st.chat_input("Ask a question about the dataset")
    if 'suggested_prompt' in st.session_state and st.session_state.suggested_prompt:
        prompt = st.session_state.suggested_prompt
        st.session_state.suggested_prompt = None
    new_question = False
    if prompt:
        user_message = Message(
            role="user",
            content=prompt,
            message_type=MessageType.USER_FACING
        ).model_dump()
        st.session_state.chat_turns.append({
            "user": user_message,
            "thinking": [],
            "assistant": None,
            "duration": None
        })
        new_question = True

    # Display chat history as grouped turns
    for turn in st.session_state.chat_turns:
        display_message(turn["user"])
        display_thinking_messages(
            turn.get("thinking", []),
            turn.get("duration")
        )
        if turn["assistant"]:
            display_message(turn["assistant"])

    # If a new question was just added, process the agent response and update the last turn
    if new_question:
        with st.spinner('The agent is deep in thoughts... and possibly snacking. Hang tight!'):
            all_prev_msgs = [msg for turn in st.session_state.chat_turns[:-1] for msg in [turn["user"]] + turn.get("thinking", []) + ([turn["assistant"]] if turn["assistant"] else [])]
            start_time = time.monotonic()
            response, updated_history = st.session_state.agent.ask(prompt, all_prev_msgs, session_id=current_session_id)
            duration = time.monotonic() - start_time
            user_indices = [i for i, msg in enumerate(updated_history) if msg["role"] == "user"]
            last_user_idx = user_indices[-1] if user_indices else 0
            thinking_msgs = updated_history[last_user_idx+1:-1] if updated_history else []
            assistant_msg = updated_history[-1] if updated_history else None
            st.session_state.chat_turns[-1]["thinking"] = thinking_msgs
            st.session_state.chat_turns[-1]["assistant"] = assistant_msg
            st.session_state.chat_turns[-1]["duration"] = duration
            st.rerun()

if __name__ == "__main__":
    main()
