"""Constrained tool-selection logic for the RAG assistant."""

import logging
from typing import Literal

from pydantic import BaseModel

from app.generate import select_tool_with_model


logger = logging.getLogger(__name__)


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Required for all knowledge-seeking questions, including policy, "
                "process, technical, troubleshooting, or factual questions."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_categories",
            "description": (
                "Use only when the user explicitly asks which category labels exist "
                "or asks to list available categories."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

class AgentDecision(BaseModel):
    # choose_tool() can only return one of these two permitted tool names.
    tool: Literal["search_knowledge_base", "list_categories"]
    selection_mode: Literal["llm", "deterministic_fallback"]


def choose_tool_fallback(question: str) -> AgentDecision:
    """Use deterministic routing if the model cannot return a valid tool call."""
    normalized_question = question.casefold()
    category_phrases = (
        "which categories",
        "what categories",
        "list categories",
        "available categories",
    )

    if any(phrase in normalized_question for phrase in category_phrases):
        return AgentDecision(
            tool="list_categories",
            selection_mode="deterministic_fallback",
        )

    return AgentDecision(
        tool="search_knowledge_base",
        selection_mode="deterministic_fallback",
    )


def choose_tool(question: str) -> AgentDecision:
    """Use the LLM to choose one validated tool, with deterministic fallback."""
    try:
        tool_name = select_tool_with_model(question, TOOL_DEFINITIONS)
        return AgentDecision(tool=tool_name, selection_mode="llm")
    except Exception as error:
        logger.warning("agent_tool_selection_failed fallback=deterministic error=%s", error)
        return choose_tool_fallback(question)
