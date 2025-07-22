# Bitext Support Sidekick

## Overview

This project implements an intelligent agent that can answer questions about customer service data through a conversational interface. The agent uses LangGraph for structured workflow management and can handle both structured queries (e.g., "What are the most frequent categories?") and unstructured analysis (e.g., "Summarize Category X").

## Features

- 🤖 Interactive chat interface built with Streamlit
- 🔄 LangGraph-based workflow with specialized nodes:
  - **Query Classification** – automatically determines query type
  - **Structured Agent** – handles direct, specific queries
  - **Unstructured Agent** – handles analysis and summary queries
  - **Out-of-Scope Handler** – manages off-topic questions
  - **Summary Node** – updates and stores user memory after each turn
  - **Next Query Recommender** – suggests relevant next queries based on user profile and conversation
- 📊 Data analysis capabilities:
  - Category analysis and distribution
  - Intent analysis
  - Semantic search
  - Exact search
  - Data aggregation
  - Common questions identification
  - Calculator
- 🚦 Automatic scope checking to filter out-of-topic questions
- 💡 Transparent reasoning with expandable thinking steps
- 🛠️ Tool-based architecture for modular functionality
- 🧠 Conversation memory and session management (multi-conversation, persistent)
- 📝 Summarized user memory (profile, preferences, interests)
- 🧭 Next Query Recommender (interactive, LLM-powered)

> **Note:** This implementation focuses on the core LangGraph requirements and new bonus features. The original reactive/plan thinking modes from part 1 have been replaced with the more sophisticated LangGraph workflow architecture.

## Architecture & DDD Structure

The application follows a modular, domain-driven architecture:

```
bitext_support_sidekick/
│
├── app.py                  # Streamlit UI (entry point)
├── agent.py                # Agent orchestrator (LangGraph interface)
├── user_memory.json        # Human-readable user memory
│
├── brain/                  # Domain logic (agent brain, nodes, workflow)
│   ├── classifier.py       # Query classification node
│   ├── graph.py            # LangGraph workflow definition
│   ├── structured_agent.py # Structured query node
│   ├── unstructured_agent.py # Unstructured query node
│   ├── out_of_scope.py     # Out-of-scope handler node
│   ├── summary.py          # Summarized memory node
│   ├── recommender.py      # Next query recommender node
│   ├── memory_manager.py   # User memory file I/O
│   ├── final_response.py   # Final response formatting
│   ├── ...
│
├── chat/                   # Messaging and LLM service abstraction
│   ├── message.py          # Message model and helpers
│   ├── service.py          # LLM API wrapper
│
├── tools/                  # Data analysis tools (domain-specific)
│   ├── aggregator.py, calculator.py, ...
│   ├── tools.py            # Tool registry/schema
│
├── scope_checker/          # Scope checking logic
│   ├── checker.py
│   ├── scope.py
│
├── assets/                 # Diagrams, images
├── notebooks/              # Jupyter notebooks (for dev/testing)
├── requirements.txt
├── README.md
```

- **No technology-leakage:** No folders like `llm/`, `langgraph/`, or `openai/` at the top level.
- **Domain-driven:** Each folder/module is named for its business role.

## Next Query Recommender (Bonus Feature)

- The agent can proactively suggest relevant next queries based on your user profile and conversation history.
- Suggestions appear as clickable buttons in the chat UI.
- You can accept, modify, or ask for more suggestions interactively.
- Example:
  - User: "Advise me what to query next"
  - Agent: [suggests 2–3 queries as buttons]
  - User: [clicks a suggestion]
  - Agent: [executes the query and returns the answer]

## Quickstart

1. **Install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Set your OpenAI API key:**
   ```bash
   export OPENAI_API_KEY=sk-...
   ```
3. **Run the app:**
   ```bash
   streamlit run app.py
   ```
4. **Test features:**
   - Try structured queries: "What are the most frequent categories?"
   - Try unstructured queries: "Summarize the ACCOUNT category."
   - Try follow-up queries: "Show me more examples."
   - Try memory: "What do you remember about me?"
   - Try recommendations: "What should I ask next?"

## Requirements

- Python 3.10 or higher
- Key dependencies:
  - LangGraph
  - LangChain
  - OpenAI
  - Streamlit
  - Pydantic

## Extending the Agent

- **Add new tools:** Implement a new function in `tools/` and register it in `tools.py`.
- **Add new agent nodes:** Create a new node in `brain/` and add it to the workflow in `graph.py`.
- **Customize user memory:** Edit `brain/summary.py` and `brain/memory_manager.py`.

## Clean Code & DDD Principles

- All code is organized by business domain, not technology.
- Naming is clear, descriptive, and domain-focused.
- Each node and major function is documented with a docstring.
- The user memory file is human-readable and self-explanatory.

---

**You’re ready to submit!** 
