from typing import Dict, Any
import pandas as pd

from bitext.datastore import _store

def exact_search(text: str, column: str | None = None, k: int = 5) -> pd.DataFrame:
    """
    Search for exact text matches in specified column(s). Case-insensitive matching.
    
    Args:
        text: Text to search for
        column: Column to search in (any column name from the dataset, or omit to search both 'instruction' and 'response')
        k: Maximum number of results to return
        
    Returns:
        pd.DataFrame: DataFrame containing the matching entries
    """
    if column is None:
        # Search in both instruction and response columns
        mask = _store.df['instruction'].str.contains(text, case=False) | _store.df['response'].str.contains(text, case=False)
    elif column not in _store.df.columns:
        raise ValueError(f"Column '{column}' not found in dataset. Available columns: {_store.df.columns.tolist()}")
    else:
        # Search in specified column only
        mask = _store.df[column].str.contains(text, case=False)
        
    return _store.df[mask].head(k).reset_index(drop=True)

def _df_to_json(df, limit=10):
    return df.head(limit).to_dict(orient="records")

TOOL_FUNC = {
    "exact_search": (
        lambda text, column=None, k=5: _df_to_json(exact_search(text, column, k)),
        {
            "name": "exact_search",
            "description": (
                "Search for exact text matches in specified column(s). Case-insensitive matching. "
                "Use this tool when you need to find entries containing specific text or phrases. "
                "Unlike semantic search, this looks for literal text matches rather than conceptual similarity. "
                "You can search in any column of the dataset, or omit the column parameter to search in both 'instruction' and 'response' columns. "
                "Example use cases: finding specific keywords, phrases, or exact text patterns in any column of the data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to search for"
                    },
                    "column": {
                        "type": "string",
                        "description": "Column to search in (any column name from the dataset, or omit to search both 'instruction' and 'response')"
                    },
                    "k": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 5
                    }
                },
                "required": ["text"]
            }
        }
    )
} 