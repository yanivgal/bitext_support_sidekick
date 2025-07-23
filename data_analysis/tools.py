from . import dataset_info
from . import data_slicing_tools as data_slicer
from . import aggregation_tools as aggregator
from . import search_tools as exact_search
from . import semantic_search
from . import analysis_tools as find_common_questions
from . import calculator

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