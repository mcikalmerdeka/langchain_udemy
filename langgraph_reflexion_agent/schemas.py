from typing import TypedDict, List, Optional
from pydantic import BaseModel, Field

# Define the reflection schema output
class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing")
    superfluous: str = Field(description="Critique of what is superfluous") # Note: superfluous is a word that means "excessive" or "unnecessary"

# Define the answer schema output
class AnswerQuestion(BaseModel):
    """Answer to the user's question"""
    
    answer: str = Field(description="~250 words detailed answer to the question")
    reflection: Optional[Reflection] = Field(default=None, description="Your reflection on the initial answer")
    search_queries: List[str] = Field(
        default_factory=list,
        description="1-3 search queries for researching improvements to address the critique of your current answer"
    )

# Define the revise answer schema output
class ReviseAnswer(AnswerQuestion): # inherit from the AnswerQuestion schema
    """Revise your original answer to your question using the new information"""
    references: List[str] = Field(
        description="Citations motivating your revised answer"
    )