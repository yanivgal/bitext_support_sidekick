from typing import Dict, Any
import pandas as pd

from bitext.datastore import _store, _CATEGORY_COL, _INTENT_COL, _FLAGS_COL, _INSTRUCTION_COL, _RESPONSE_COL

def dataset_info() -> Dict[str, Any]:
    """
    Get comprehensive information about the customer support dataset.
    
    Returns:
        Dict containing:
        - dataset: General dataset information
            - total_entries: Total number of entries
            - columns: List of all columns in the dataset
            - description: Dataset overview
                - purpose: Main purpose of the dataset
                - content: Description of the data content
                - use_cases: List of analytical use cases
        - instruction: Statistics about customer instructions
            - length: Statistics about instruction lengths (mean, median, min, max)
        - response: Statistics about agent responses
            - length: Statistics about response lengths (mean, median, min, max)
        - category: Statistics about categories
            - total: Number of unique categories
            - distribution: Full distribution of categories
        - intent: Statistics about intents
            - total: Number of unique intents
            - distribution: Full distribution of intents
        - flags: Statistics about flags
            - total: Number of unique flags
            - distribution: Full distribution of flags
    """
    category_dist = _store.df[_CATEGORY_COL].value_counts()
    intent_dist = _store.df[_INTENT_COL].value_counts()
    flag_dist = _store.df[_FLAGS_COL].value_counts()
    
    instruction_lengths = _store.df[_INSTRUCTION_COL].str.len()
    response_lengths = _store.df[_RESPONSE_COL].str.len()
    
    return {
        "dataset": {
            "total_entries": int(len(_store.df)),
            "columns": _store.get_columns(),
            "description": {
                "purpose": "A dataset for analyzing customer service interaction patterns, understanding query distributions, and studying the relationships between customer intents, categories, and response characteristics",
                "content": "Contains customer queries, agent responses, and metadata including categories, intents, and flags",
                "use_cases": [
                    "Analyzing customer service interaction patterns",
                    "Understanding query distributions and common scenarios",
                    "Studying relationships between intents and categories",
                    "Evaluating response characteristics and effectiveness"
                ]
            }
        },
        "instruction": {
            "length": {
                "mean": float(instruction_lengths.mean()),
                "median": float(instruction_lengths.median()),
                "min": int(instruction_lengths.min()),
                "max": int(instruction_lengths.max())
            }
        },
        "response": {
            "length": {
                "mean": float(response_lengths.mean()),
                "median": float(response_lengths.median()),
                "min": int(response_lengths.min()),
                "max": int(response_lengths.max())
            }
        },
        "category": {
            "total": int(len(category_dist)),
            "distribution": {str(k): int(v) for k, v in category_dist.to_dict().items()}
        },
        "intent": {
            "total": int(len(intent_dist)),
            "distribution": {str(k): int(v) for k, v in intent_dist.to_dict().items()}
        },
        "flags": {
            "total": int(len(flag_dist)),
            "distribution": {str(k): int(v) for k, v in flag_dist.to_dict().items()}
        }
    }

TOOL_FUNC = {
    "dataset_info": (
        dataset_info,
        {
            "name": "dataset_info",
            "description": (
                "Get comprehensive information about the dataset including its purpose, content, use cases, "
                "and key features. Also includes columns, categories, intents, their distributions, and "
                "basic statistics. Use this to understand the dataset structure and content before "
                "performing specific analyses."
            ),
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    )
} 