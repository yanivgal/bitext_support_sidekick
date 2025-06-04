# Bitext Support Sidekick 🤖

A Streamlit-based chat application that helps users analyze and understand the Bitext Customer Service Tagged Training dataset through natural language interactions.

## Overview

This project implements an intelligent agent that can answer questions about customer service data through a conversational interface. The agent can handle both structured queries (e.g., "What are the most frequent categories?") and unstructured analysis (e.g., "Summarize Category X").

## Features

- 🤖 Interactive chat interface built with Streamlit
- 🔄 Two agent modes:
  - Reactive: Step-by-step thinking and execution
  - Plan: Creates a structured plan before execution
- 📊 Data analysis capabilities:
  - Category analysis and distribution
  - Intent analysis
  - Semantic search
  - Exact search
  - Data aggregation
  - Common questions identification
- 💡 Transparent reasoning with expandable thinking steps
- 🛠️ Tool-based architecture for modular functionality

## Architecture

The application follows a modular architecture:

- `app.py`: Main Streamlit application handling UI and chat flow
- `agent.py`: Core agent implementation with two modes of operation
- `tools/`: Collection of specialized tools for data analysis:
  - `data_slicer.py`: Data filtering and slicing
  - `find_common_questions.py`: Common question analysis
  - `aggregator.py`: Data aggregation functions
  - `exact_search.py`: Exact text search
  - `semantic_search.py`: Semantic search capabilities
  - `dataset_info.py`: Dataset metadata and information
  - `calculator.py`: Mathematical operations

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

- Python 3.10 or higher is required for this project
- The project uses type hints and modern Python features that are not available in earlier versions
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
├── app.py                 # Main Streamlit application
├── agent.py              # Core agent implementation
├── system_prompts.py     # System prompts for different modes
├── tools/                # Data analysis tools
│   ├── data_slicer.py
│   ├── find_common_questions.py
│   ├── aggregator.py
│   ├── exact_search.py
│   ├── semantic_search.py
│   ├── dataset_info.py
│   └── calculator.py
├── message_models/       # Message type definitions
├── notebooks/           # Jupyter notebooks for development
└── requirements.txt     # Project dependencies
```

## Contributing

Feel free to submit issues and enhancement requests! 