from data_analysis import dataset_info
from data_analysis import data_slicing_tools
from data_analysis import aggregation_tools
from data_analysis import search_tools
from data_analysis import analysis_tools

_TOOL_FUNCS = {}

_TOOL_FUNCS.update(dataset_info.TOOL_FUNC)
_TOOL_FUNCS.update(data_slicing_tools.TOOL_FUNC)
_TOOL_FUNCS.update(aggregation_tools.TOOL_FUNC)
_TOOL_FUNCS.update(search_tools.TOOL_FUNC)
_TOOL_FUNCS.update(analysis_tools.TOOL_FUNC)

TOOLS_SCHEMA = [
    {"type": "function", "function": meta} for _, meta in _TOOL_FUNCS.values()
]