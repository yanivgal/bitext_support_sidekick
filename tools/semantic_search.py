from typing import Dict, Any
import pandas as pd

from bitext.datastore import _store

def semantic_search(text: str, k: int = 5) -> pd.DataFrame:
    """
    Perform semantic search on the dataset using sentence embeddings.
    
    Args:
        text: The query text to search for
        k: Number of most similar results to return
        
    Returns:
        pd.DataFrame: DataFrame containing the k most semantically similar entries
    """
    q = _store.model.encode([text])
    idx = _store.nn.kneighbors(q, n_neighbors=k, return_distance=False)[0]
    return _store.df.iloc[idx].reset_index(drop=True)

def _df_to_json(df, limit=10):
    return df.head(limit).to_dict(orient="records")

TOOL_FUNC = {
    "semantic_search": (
        lambda text, k=5: _df_to_json(semantic_search(text, k)),
        {
            "name": "semantic_search",
            "description": (
                "Perform semantic search on the dataset using sentence embeddings. "
                "This tool finds the most semantically similar entries to the given query text. "
                "Use this when you need to find entries that are conceptually similar to a given text, "
                "even if they don't contain the exact same words. "
                "Example use cases: finding similar customer questions, related support requests, "
                "or semantically similar responses."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The query text to search for semantically similar entries"
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of most similar results to return",
                        "default": 5
                    }
                },
                "required": ["text"]
            }
        }
    )
} 