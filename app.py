import streamlit as st
from agent import Agent
from chat.message import MessageType, Message
import time

from dotenv import load_dotenv

load_dotenv()


def _format_duration(seconds: float) -> str:
    """Return a human friendly duration string."""
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
    """Display a group of thinking messages in a single collapsible section."""
    if not messages:
        return

    if duration is not None:
        time_str = _format_duration(duration)
        label = f"🤔 Thought for {time_str}"
    else:
        label = "Thinking...  🤔"

    with st.expander(label):
        
        # First pass: collect all tool calls and their indices
        tool_calls = []
        for i, msg in enumerate(messages):
            if msg["message_type"] == MessageType.TOOL_CALL and "tool_calls" in msg:
                for tool_call in msg["tool_calls"]:
                    tool_calls.append((i, tool_call))
        
        # Second pass: display messages in order
        current_tool_call_idx = 0
        for i, msg in enumerate(messages):
            if msg["message_type"] == MessageType.THINKING:
                st.write("🤔  " + msg["content"])
                if "reasoning" in msg:
                    st.markdown(f"> **Reasoning:** {msg['reasoning']}")
            elif msg["message_type"] == MessageType.TOOL_CALL:
                # Only display the initial message if there are tool calls
                if tool_calls and current_tool_call_idx == 0:
                    st.write(f"🔧  {msg['content']}")
            elif msg["message_type"] == MessageType.TOOL_RESULT:
                # If we have a pending tool call, display it as a child of the main message
                if current_tool_call_idx < len(tool_calls):
                    tool_call_idx, tool_call = tool_calls[current_tool_call_idx]
                    st.markdown(f"> 🛠️  Calling {tool_call['function']['name']} tool with args: {tool_call['function']['arguments']}")
                    current_tool_call_idx += 1
                # Then display the result
                st.markdown(f"""> ✅  Tool Result:  
                            {msg['content']}
                            """)

def display_message(message):
    """Display a user-facing message in the chat."""
    if message["role"] == "system":
        return
    if message["message_type"] == MessageType.USER_FACING:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant" and "reasoning" in message:
                with st.expander("Agent's Reasoning"):
                    st.write(message["reasoning"])

def main():
    st.title("🤖 Bitext Support Sidekick")
    st.caption("Your friendly neighborhood data detective! I'll help you crack the case of customer conversations, decode intents, and make your support experience less 'support-ive' and more 'awesome-ive'! 🕵️‍♂️")
    
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
    
    # Chat input
    prompt = st.chat_input("Ask a question about the dataset")
    new_question = False
    if prompt:
        # User message
        user_message = Message(
            role="user",
            content=prompt,
            message_type=MessageType.USER_FACING
        ).model_dump()
        # Add a new turn immediately with only the user message
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
            # Get agent's response (returns full updated history)
            all_prev_msgs = [msg for turn in st.session_state.chat_turns[:-1] for msg in [turn["user"]] + turn.get("thinking", []) + ([turn["assistant"]] if turn["assistant"] else [])]
            start_time = time.monotonic()
            response, updated_history = st.session_state.agent.ask(prompt, all_prev_msgs)
            duration = time.monotonic() - start_time
            # Extract new thinking messages and assistant answer
            user_indices = [i for i, msg in enumerate(updated_history) if msg["role"] == "user"]
            last_user_idx = user_indices[-1] if user_indices else 0
            thinking_msgs = updated_history[last_user_idx+1:-1]
            assistant_msg = updated_history[-1]
            # Update the last turn
            st.session_state.chat_turns[-1]["thinking"] = thinking_msgs
            st.session_state.chat_turns[-1]["assistant"] = assistant_msg
            st.session_state.chat_turns[-1]["duration"] = duration
            st.rerun()

if __name__ == "__main__":
    main()
