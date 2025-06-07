# Bitext Support Sidekick 🤖

A Streamlit-based chat application that helps users analyze and understand the Bitext Customer Service Tagged Training dataset through natural language interactions. The dataset is automatically downloaded from HuggingFace on first run and cached locally.

## Overview

This project implements an intelligent agent that can answer questions about customer service data through a conversational interface. The agent can handle both structured queries (e.g., "What are the most frequent categories?") and unstructured analysis (e.g., "Summarize Category X").

## Features

- 🤖 Interactive chat interface built with Streamlit
- 🔄 Two agent modes:
  - **Reactive** – step-by-step thinking and execution
  - **Plan** – creates a structured plan before execution
- 📊 Data analysis capabilities:
  - Category analysis and distribution
  - Intent analysis
  - Semantic search
  - Exact search
  - Data aggregation
  - Common questions identification
- 🚦 Automatic scope checking to filter out-of-topic questions
- 💡 Transparent reasoning with expandable thinking steps
- 🛠️ Tool-based architecture for modular functionality

## Architecture

The application follows a modular architecture:

- `app.py` – Streamlit UI and chat flow
- `agent.py` – orchestrates the conversation, scope checking and reasoning
- `brain/` – planning and reactive strategies
- `chat/` – message models and wrapper around the OpenAI API
- `bitext/datastore.py` – loads the dataset and builds the search index
- `scope_checker/` – verifies if a question is in scope
- `tools/` – data analysis tools:
  - `data_slicer.py` – filter/group/sort the data
  - `find_common_questions.py` – discover frequent question patterns
  - `aggregator.py` – aggregation functions
  - `exact_search.py` – literal text search
  - `semantic_search.py` – embedding based search
  - `dataset_info.py` – dataset metadata
  - `calculator.py` – numerical calculations

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Run the application:
```bash
streamlit run app.py
```

## Requirements

- Python 3.10 or higher is required for this project. The code relies on
  [PEP&nbsp;604](https://peps.python.org/pep-0604/) union types (e.g. `list | None`),
  which are not available in earlier Python versions.
- We recommend using a virtual environment to manage dependencies

## Usage

1. Select your preferred agent mode in the sidebar:
   - Reactive: For step-by-step thinking and execution
   - Plan: For structured planning before execution

2. Ask questions about the dataset in natural language:
   - "What are the most frequent categories?"
   - "Show examples of Category X"
   - "Summarize how agents respond to Intent Y"

3. View the agent's thinking process by expanding the "Thinking..." sections

## Project Structure

```
.
├── app.py              # Streamlit entry point
├── agent.py            # Conversation orchestration
├── brain/              # Planning and reactive logic
├── chat/               # Message models and OpenAI wrapper
├── bitext/
│   └── datastore.py    # Dataset loader and embedding index
├── scope_checker/      # Out-of-scope detection
├── tools/              # Data analysis tools
│   ├── data_slicer.py
│   ├── find_common_questions.py
│   ├── aggregator.py
│   ├── exact_search.py
│   ├── semantic_search.py
│   ├── dataset_info.py
│   └── calculator.py
├── notebooks/          # Development notebooks
└── requirements.txt    # Project dependencies
```

## Contributing

Feel free to submit issues and enhancement requests! 
