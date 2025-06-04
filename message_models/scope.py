from enum import Enum
from pydantic import BaseModel, Field

class ScopeEnum(str, Enum):
    IN_SCOPE = "in_scope"
    OUT_OF_SCOPE = "out_of_scope"

class ScopeCheck(BaseModel):
    reasoning: str = Field(..., description="A short explanation of the reasoning behind the scope check")
    scope: ScopeEnum = Field(..., description="The scope of the user's request")
