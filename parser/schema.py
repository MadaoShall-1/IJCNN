# parser/schema.py

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ParsedQuestion(BaseModel):
    domain: str = Field(default="physics")
    topic: str = Field(default="unknown")
    subtopic: str = Field(default="unknown")
    question_type: str = Field(default="calculation")

    target_quantity: str = Field(default="unknown")
    known_variables: Dict[str, str] = Field(default_factory=dict)
    unknown_variables: List[str] = Field(default_factory=list)

    answer_type: str = Field(default="unknown")
    unit_expected: str = Field(default="unknown")

    requires_diagram_reasoning: bool = False
    requires_formula_retrieval: bool = True

    answer_options: Optional[Dict[str, str]] = None

    implicit_conditions: List[str] = Field(default_factory=list)
    physical_relations: List[str] = Field(default_factory=list)

    parser_confidence: float = 0.0