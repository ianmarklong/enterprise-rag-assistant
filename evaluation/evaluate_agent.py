"""Evaluate LLM tool selection and the deterministic fallback."""

import json
from pathlib import Path
from unittest.mock import patch

from app.agent import choose_tool


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "agent_questions.json"


def load_questions(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_tool_selection(questions: list[dict]) -> None:
    correct_choices = 0
    fallback_count = 0

    for item in questions:
        decision = choose_tool(item["question"])
        correct = decision.tool == item["expected_tool"]
        correct_choices += correct
        fallback_count += decision.selection_mode == "deterministic_fallback"

        print(f"Question: {item['question']}")
        print(f"Expected tool: {item['expected_tool']}")
        print(f"Selected tool: {decision.tool}")
        print(f"Selection mode: {decision.selection_mode}")
        print(f"Correct: {correct}\n")

    print("AGENT EVALUATION")
    print("-" * 50)
    print(f"Questions: {len(questions)}")
    print(f"Tool-selection accuracy: {correct_choices / len(questions):.2%}")
    print(f"Fallback usage: {fallback_count / len(questions):.2%}")


def evaluate_fallback() -> None:
    with patch(
        "app.agent.select_tool_with_model",
        side_effect=RuntimeError("simulated model failure"),
    ):
        decision = choose_tool("What categories are available?")

    passed = (
        decision.tool == "list_categories"
        and decision.selection_mode == "deterministic_fallback"
    )
    print(f"Fallback test passed: {passed}")


def main() -> None:
    evaluate_tool_selection(load_questions(QUESTIONS_PATH))
    evaluate_fallback()


if __name__ == "__main__":
    main()
