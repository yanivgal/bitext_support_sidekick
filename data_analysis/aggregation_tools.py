from typing import Dict, Any, List, Union
import pandas as pd

from bitext.datastore import _store

_df = _store.df

def aggregator(
    group_by: str | List[str],
    metrics: List[str] = ["count"],
    filters: Dict[str, Any] = None,
    sort_by: str = None,
    limit: int = 20
) -> Dict[str, Any]:
    """
    Perform flexible aggregations on the dataset.
    
    Args:
        group_by: Column(s) to group by (e.g., "category", ["category", "intent"])
        metrics: List of metrics to calculate:
            - "count": Count of rows
            - "percentage": Percentage of total
            - "unique": Count of unique values in another column
            - "text_stats": Basic stats about text columns (avg length, common words)
        filters: Optional filters to apply (e.g., {"category": "ACCOUNT"})
        sort_by: Column to sort results by
        limit: Maximum number of results to return
    
    Returns:
        Dict containing:
        - results: The aggregated data
        - metadata: Information about the aggregation
    """
    df = _df.copy()
    
    # Apply filters if any
    if filters:
        for col, val in filters.items():
            df = df[df[col] == val]
    
    # Handle group_by
    if isinstance(group_by, str):
        group_by = [group_by]
    
    # Perform aggregation
    results = []
    for group in df.groupby(group_by):
        group_data = {
            "group": dict(zip(group_by, group[0])),
            "metrics": {}
        }
        
        group_df = group[1]
        
        # Calculate requested metrics
        if "count" in metrics:
            group_data["metrics"]["count"] = len(group_df)
        
        if "percentage" in metrics:
            group_data["metrics"]["percentage"] = len(group_df) / len(df) * 100
        
        if "unique" in metrics:
            for col in df.columns:
                if col not in group_by:
                    group_data["metrics"][f"unique_{col}"] = group_df[col].nunique()
        
        if "text_stats" in metrics:
            for col in ["instruction", "response"]:
                if col in df.columns:
                    group_data["metrics"][f"{col}_stats"] = {
                        "avg_length": group_df[col].str.len().mean(),
                        "word_count": group_df[col].str.split().str.len().mean(),
                        "common_words": group_df[col].str.split().explode().value_counts().head(5).to_dict()
                    }
        
        results.append(group_data)
    
    # Sort results if requested
    if sort_by:
        results.sort(key=lambda x: x["metrics"].get(sort_by, 0), reverse=True)
    
    # Apply limit
    results = results[:limit]
    
    return {
        "results": results,
        "metadata": {
            "total_groups": len(results),
            "total_rows": len(df),
            "group_by": group_by,
            "metrics": metrics
        }
    }

TOOL_FUNC = {
    "aggregator": (
        aggregator,
        {
            "name": "aggregator",
            "description": (
                "Perform flexible aggregations on the dataset. "
                "Use this tool when you need to analyze data by grouping it and calculating various metrics. "
                "Supports counting rows, calculating percentages, counting unique values, and analyzing text statistics. "
                "Example use cases: counting entries by category, calculating percentage distribution of intents, "
                "or analyzing text length patterns in customer messages."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "group_by": {
                        "type": ["string", "array"],
                        "description": "Column(s) to group by. Can be a single column name or a list of column names. Example: 'category' or ['category', 'intent']",
                        "items": {
                            "type": "string"
                        }
                    },
                    "metrics": {
                        "type": "array",
                        "description": "List of metrics to calculate. Available metrics: 'count', 'percentage', 'unique', 'text_stats'",
                        "items": {
                            "type": "string",
                            "enum": ["count", "percentage", "unique", "text_stats"]
                        },
                        "default": ["count"]
                    },
                    "filters": {
                        "type": "object",
                        "description": "Optional filters to apply. Example: {'category': 'ACCOUNT'}"
                    },
                    "sort_by": {
                        "type": "string",
                        "description": "Column to sort results by"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10
                    }
                },
                "required": ["group_by"]
            }
        }
    )
}
