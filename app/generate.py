import ollama
import os


MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct-2507-q4_K_M")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Keep the network location outside the code so local and Docker runs can
# point at their appropriate Ollama service without a code change.
ollama_client = ollama.Client(host=OLLAMA_HOST)


def generate_answer(prompt):
    response = ollama_client.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ],
        keep_alive="30m",
        options={
            "temperature": 0,
            "num_predict": 180
        }
    )

    answer = response["message"].get("content", "").strip()
    if not answer:
        raise RuntimeError("Ollama returned an empty answer")

    return answer


def select_tool_with_model(question, tools):
    """Ask Ollama to choose exactly one tool from the supplied definitions."""
    response = ollama_client.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a constrained enterprise assistant. Select exactly one "
                    "tool and do not answer the user directly. Use list_categories "
                    "only when the user explicitly asks to list or identify the "
                    "available category labels. For every question seeking knowledge, "
                    "instructions, policy, troubleshooting, or facts, use "
                    "search_knowledge_base."
                ),
            },
            {"role": "user", "content": question},
        ],
        tools=tools,
        think=False,
        keep_alive="30m",
        options={
            "temperature": 0,
            "num_predict": 60,
        },
    )

    tool_calls = response["message"].get("tool_calls", [])
    if len(tool_calls) != 1:
        raise RuntimeError("Ollama did not select exactly one tool")

    tool_name = tool_calls[0]["function"].get("name")
    if not tool_name:
        raise RuntimeError("Ollama returned a tool call without a name")

    return tool_name
