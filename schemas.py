# We want to ensure that the output we get from the LLM is in a structured format,
# so we want the format to be with a response field that is having the original essay,
# we want the critique field, which is having the critique for that essay, and we want a search field,
# which will be a list of values that we should search for.
# And for that, we're going to leverage function calling and specifically function calling that would make sure
# that the output format of the LLM is going to be in an object we create which is AnswerQuestion class

from typing import List

from pydantic import BaseModel, Field


class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous") # superfluous means that information that add up no value.


class AnswerQuestion(BaseModel):
    """Answer the question."""

    answer: str = Field(description="~250 word detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    search_queries: List[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )


class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question."""

    references: List[str] = Field(
        description="Citations motivating your updated answer."
    )

