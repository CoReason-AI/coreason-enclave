from pydantic import BaseModel, Field


class UserContext(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    username: str = Field(..., description="Username")
    privacy_budget_spent: float = Field(0.0, description="Amount of privacy budget spent")
    privacy_budget_limit: float = Field(10.0, description="Total privacy budget limit")
