from pydantic import BaseModel, Field
from typing import List, Dict, Any

_system_prompt = (
    "Create a structured plan to answer the user's question. "
    "Break down the work into clear steps, considering dependencies between steps. "
    "For each step, specify what needs to be done and what we expect to get from it."
)

class PlanningStep(BaseModel):
    reasoning: str = Field(
        description="Explanation of why this step is needed and how it contributes to the overall goal"
    )
    action: str = Field(
        description="What needs to be done in this step"
    )
    expected_result: str = Field(
        description="What we expect to get from this step"
    )
    depends_on: List[int] = Field(
        description="Indices of steps this step depends on (empty list if no dependencies)",
        default_factory=list
    )

class PlanningThinking(BaseModel):
    steps: List[PlanningStep] = Field(
        description="Sequence of steps needed to achieve the goal"
    )
    goal: str = Field(
        description="The overall goal we're trying to achieve"
    )