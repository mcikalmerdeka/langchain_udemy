"""
Output parsers for converting LLM string outputs to structured formats.

This module provides Pydantic-based parsers to convert unstructured LLM responses
into structured data objects that can be used by other systems.
"""

from typing import List, Dict, Any
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Create a pydantic object
class Summary(BaseModel):
    """
    A summary of a person's life. This is a pydantic object that will be used to parse the output of the LLM.

    Attributes:
        summary: A short summary of the person's life.
        facts: Two interesting facts about the person.
    """
    # Fields
    summary: str = Field(description="summary")
    facts: List[str] = Field(description="interesting facts about them")

    # Method to convert the pydantic object to a dictionary
    def to_dict(self) -> Dict[str, Any]:
        return {"summary": self.summary, "facts": self.facts}

# Create a summary parser object
summary_parser = PydanticOutputParser(pydantic_object=Summary)