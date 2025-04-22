from pydantic import BaseModel, Field
from typing import Optional, List
from typing_extensions import TypedDict

class ThoughtOutput(BaseModel):
    subtasks: List[str] = Field(description="A list of detailed, logically separated subtasks derived from the user query. Each string should represent a standalone, actionable instruction.")

class ScoredAnswer(BaseModel):
    answer: str = Field(description="Potential answer to the query")
    score: int = Field(description="Score from 1-10 for helpfulness and accuracy")

class ScoredAnswerOutput(BaseModel):
    answer_1: ScoredAnswer = Field(description="First potential answer with score")
    answer_2: ScoredAnswer = Field(description="Second potential answer with score")

class ReflectionOutput(BaseModel):
    score: int = Field(description="Quality score from 1-10")
    feedback: str = Field(description="Brief improvement suggestion")

class ToolSuggestionOutput(BaseModel):
    tools_needed: List[str] = Field(description="List of relevant tools needed", default_factory=list)