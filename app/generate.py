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
