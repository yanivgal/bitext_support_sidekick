from typing import Dict, Any
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

from bitext.datastore import _store

_df = _store.df
_model = _store.model

def find_common_questions(filter: Dict[str, Any] | None = None, text_field: str = "instruction", n: int = 10) -> Dict[str, Any]:
        """
        Analyzes customer messages to find common types of requests and questions.
        Groups similar customer inquiries together, showing you:
        - The most common ways customers ask for help
        - How many customers ask similar questions
        - Real examples of how customers phrase their requests

        This helps understand what customers need help with most often and how they typically ask for it.
        
        Args:
            filter: Optional dictionary of column-value pairs to filter the data (e.g., {"category": "ACCOUNT", "intent": "cancel_order"})
            text_field: Which part to analyze - customer questions ('instruction') or agent responses ('response')
            n: How many common patterns to show
        
        Returns:
            Dict containing:
            - patterns: List of common customer requests found, each with:
                - pattern: The most typical way this request is phrased
                - count: How many customers asked similar questions
                - examples: Real examples of how customers asked this question
            - total_entries: Total number of customer messages analyzed
            - available_fields: List of fields that were available for analysis
        """
        df = _df.copy()
        
        # Apply filters if specified
        if filter is not None:
            # Validate filter keys
            invalid_keys = [key for key in filter.keys() if key not in df.columns]
            if invalid_keys:
                raise ValueError(f"Invalid filter keys: {invalid_keys}. Available columns: {df.columns.tolist()}")
            
            # Apply each filter condition
            for col, val in filter.items():
                df = df[df[col] == val]
        
        # Find suitable text columns if the specified one doesn't exist
        available_fields = df.columns.tolist()
        if text_field not in available_fields:
            # Look for columns that might contain text (string type)
            text_columns = [col for col in available_fields if df[col].dtype == 'object']
            if text_columns:
                text_field = text_columns[0]  # Use the first text column found
        
        # Get text to analyze
        texts = df[text_field].astype(str)
        
        # If we have no texts, return empty result
        if len(texts) == 0:
            return {
                "patterns": [],
                "total_entries": 0,
                "available_fields": available_fields
            }
        
        # Use the existing sentence transformer model to get embeddings
        embeddings = _model.encode(texts.tolist(), show_progress_bar=False)
        
        # Calculate number of clusters (can't be more than number of texts)
        n_clusters = min(n, len(texts))
        
        # If we have only one text, return it as a single pattern
        if n_clusters == 1:
            return {
                "patterns": [{
                    "pattern": texts.iloc[0],
                    "count": 1,
                    "examples": [texts.iloc[0]]
                }],
                "total_entries": 1,
                "available_fields": available_fields
            }
        
        # Use clustering to find patterns
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
        clusters = kmeans.fit_predict(embeddings)
        
        # Analyze each cluster
        patterns = []
        for i in range(kmeans.n_clusters):
            cluster_texts = texts[clusters == i]
            # Get most representative example (closest to cluster center)
            center = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(embeddings[clusters == i] - center, axis=1)
            example_idx = np.argmin(distances)
            example = cluster_texts.iloc[example_idx]
            
            patterns.append({
                "pattern": example,
                "count": len(cluster_texts),
                "examples": cluster_texts.sample(min(3, len(cluster_texts))).tolist()
            })
        
        # Sort patterns by count
        patterns.sort(key=lambda x: x["count"], reverse=True)
        
        return {
            "patterns": patterns[:n],
            "total_entries": len(df),
            "available_fields": available_fields
        }

TOOL_FUNC = {
    "find_common_questions": (
        find_common_questions,
        {
            "name": "find_common_questions",
            "description": (
                "Analyzes customer messages to find common patterns and group similar inquiries together. "
                "This tool helps understand how customers typically phrase their requests and what they need help with most often. "
                "It uses machine learning to identify patterns in customer messages and provides examples of how customers ask similar questions. "
                "Use this when you want to understand common customer needs or improve response templates. "
                "Note: This tool may take a few minutes to run as it performs complex clustering analysis on the data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "object",
                        "description": "Optional dictionary of column-value pairs to filter the data (e.g., {'category': 'ACCOUNT', 'intent': 'cancel_order'})"
                    },
                    "text_field": {
                        "type": "string",
                        "description": "Which part to analyze - customer questions ('instruction') or agent responses ('response')",
                        "default": "instruction"
                    },
                    "n": {
                        "type": "integer",
                        "description": "How many common patterns to show",
                        "default": 10
                    }
                }
            }
        }
    )
}