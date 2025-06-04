from tools import dataset_info
from tools import data_slicer
from tools import aggregator
from tools import exact_search
from tools import semantic_search
from tools import find_common_questions
from tools import calculator

_TOOL_FUNCS = {}

_TOOL_FUNCS.update(dataset_info.TOOL_FUNC)
_TOOL_FUNCS.update(data_slicer.TOOL_FUNC)
_TOOL_FUNCS.update(aggregator.TOOL_FUNC)
_TOOL_FUNCS.update(exact_search.TOOL_FUNC)
_TOOL_FUNCS.update(semantic_search.TOOL_FUNC)
_TOOL_FUNCS.update(find_common_questions.TOOL_FUNC)
_TOOL_FUNCS.update(calculator.TOOL_FUNC)

TOOLS_SCHEMA = [
    {"type": "function", "function": meta} for _, meta in _TOOL_FUNCS.values()
]