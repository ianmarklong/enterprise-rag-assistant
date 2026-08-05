from openai import OpenAI


MODEL_NAME = "gpt-5.6-luna"


def generate_answer(prompt):
    client = OpenAI()

    response = client.responses.create(
        model=MODEL_NAME,
        input=prompt
    )

    return response.output_text