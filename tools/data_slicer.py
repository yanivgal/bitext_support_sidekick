from typing import Dict, Any, List, Union
import pandas as pd

from bitext.datastore import _store

_df = _store.df

def _df_to_json(df, limit=10):
    return df.head(limit).to_dict(orient="records")

def data_slicer(
    filter: Dict[str, Any] | None = None,
    group_by: str | List[str] | None = None,
    sort_by: str | Dict[str, bool] | None = None,
    limit: int | None = None,
    random_sample: bool = False
) -> pd.DataFrame:
    """
    Get a slice of the dataset based on filtering, grouping, sorting, and sampling criteria.
    
    Args:
        filter: Optional dictionary of column-value pairs to filter the data.
               Values can be single values or lists for multiple matches.
               Example: {"category": "ACCOUNT"} or {"intent": ["cancel_order", "track_order"]}
        group_by: Column(s) to group by. Can be a single column name or a list of column names.
                 Example: "category" or ["category", "intent"]
        sort_by: Column to sort by. Can be a column name or a dict with column name and ascending flag.
                Example: "category" or {"category": False} for descending sort
        limit: Maximum number of rows to return
        random_sample: If True, return random rows instead of first N rows when limit is specified
    
    Returns:
        pd.DataFrame: The filtered, grouped, sorted, and sampled subset of the data
        
    Raises:
        ValueError: If invalid column names are provided in filter, group_by, or sort_by
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
            if isinstance(val, list):
                df = df[df[col].isin(val)]
            else:
                df = df[df[col] == val]
    
    # Apply grouping if specified
    if group_by is not None:
        # Convert single column to list for consistent handling
        if isinstance(group_by, str):
            group_by = [group_by]
            
        # Validate group_by columns
        invalid_cols = [col for col in group_by if col not in df.columns]
        if invalid_cols:
            raise ValueError(f"Invalid group_by columns: {invalid_cols}. Available columns: {df.columns.tolist()}")
            
        df = df.groupby(group_by).apply(lambda x: x).reset_index(drop=True)
    
    # Apply sorting if specified
    if sort_by is not None:
        if isinstance(sort_by, str):
            if sort_by not in df.columns:
                raise ValueError(f"Invalid sort_by column: {sort_by}. Available columns: {df.columns.tolist()}")
            df = df.sort_values(sort_by)
        elif isinstance(sort_by, dict):
            col, ascending = next(iter(sort_by.items()))
            if col not in df.columns:
                raise ValueError(f"Invalid sort_by column: {col}. Available columns: {df.columns.tolist()}")
            df = df.sort_values(col, ascending=ascending)
    
    # Apply sampling if limit is specified
    if limit is not None:
        if random_sample:
            df = df.sample(min(limit, len(df)))
        else:
            df = df.head(limit)
    
    return df.reset_index(drop=True)

TOOL_FUNC = {
    "data_slicer": (
        lambda filter=None, group_by=None, sort_by=None, limit=None, random_sample=False: _df_to_json(
            data_slicer(filter, group_by, sort_by, limit, random_sample)
        ),
        {
            "name": "data_slicer",
            "description": (
                "Get a slice of the dataset based on filtering, grouping, sorting, and sampling criteria. "
                "Use this tool when you need to extract specific portions of the data based on various conditions. "
                "Supports filtering by column values, grouping by columns, sorting, and sampling. "
                "Example use cases: getting all ACCOUNT category entries, grouping by intent and sorting by count, "
                "or getting a random sample of 100 rows."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "object",
                        "description": "Optional dictionary of column-value pairs to filter the data. Values can be single values or lists for multiple matches. Example: {'category': 'ACCOUNT'} or {'intent': ['cancel_order', 'track_order']}"
                    },
                    "group_by": {
                        "type": ["string", "array"],
                        "description": "Column(s) to group by. Can be a single column name or a list of column names. Example: 'category' or ['category', 'intent']",
                        "items": {
                            "type": "string"
                        }
                    },
                    "sort_by": {
                        "type": ["string", "object"],
                        "description": "Column to sort by. Can be a column name or a dict with column name and ascending flag. Example: 'category' or {'category': false} for descending sort"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of rows to return"
                    },
                    "random_sample": {
                        "type": "boolean",
                        "description": "If true, return random rows instead of first N rows when limit is specified",
                        "default": False
                    }
                }
            }
        }
    )
} 