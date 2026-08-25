import json
from pathlib import Path

from app.load import load_knowledge_base
from app.chunk import paragraph_chunk_knowledge_base
from app.embed import load_embedding_model, embed_chunks
from app.vector_store import create_vector_store, search_vector_store
from app.rerank import load_reranker, rerank
from app.prompt import build_prompt
from app.generate import generate_answer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_PATH = PROJECT_ROOT / "knowledge_base"
QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "questions.json"

def load_questions(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def citation_hit(answer, expected_sources):
    if len(expected_sources) == 0:
        return False

    for source in expected_sources:
        if source in answer:
            return True

    return False

REFUSAL_PHRASES = [
    "does not contain enough information",
    "documentation does not contain",
    "not enough information",
    "cannot answer from the provided context"
]

def detected_refusal(answer):
    return answer.strip() == "INSUFFICIENT_DOCUMENTATION"

def evaluate_answers(questions, chunks, model, client, reranker):
    answerable_count = 0    
    citation_hits = 0
    unexpected_refusals = 0

    unanswerable_count = 0
    correct_refusals = 0

    for item in questions:
        question = item["question"]
        expected_sources = item["expected_sources"]
        answerable = item["answerable"]

        query_embedding = model.encode_query(
            question,
            convert_to_numpy=True
        )

        candidates = search_vector_store(
            client,
            query_embedding,
            top_k=8
        )

        results = rerank(
            question,
            candidates,
            reranker,
            top_k=3
        )

        prompt = build_prompt(
            question,
            results
        )

        answer = generate_answer(prompt)

        refusal = detected_refusal(answer)

        print(f"\nQuestion: {question}")
        print(f"Answerable: {answerable}")
        print(f"Answer: {answer}")

        if answerable:
            answerable_count += 1

            citation = citation_hit(
                answer,
                expected_sources
            )

            if citation:
                citation_hits += 1

            if refusal:
                unexpected_refusals += 1

            print(f"Citation hit: {citation}")
            print(f"Unexpected refusal: {refusal}")

        else:
            unanswerable_count += 1

            if refusal:
                correct_refusals += 1

            print(f"Correct refusal: {refusal}")

    print("\nANSWER EVALUATION")
    print("-" * 50)

    if answerable_count > 0:
        print(
            f"Citation accuracy: "
            f"{citation_hits / answerable_count:.2%}"
        )

        print(
            f"Unexpected refusal rate: "
            f"{unexpected_refusals / answerable_count:.2%}"
        )

    if unanswerable_count > 0:
        print(
            f"Correct refusal rate: "
            f"{correct_refusals / unanswerable_count:.2%}"
        )

def main():
    documents = load_knowledge_base(KNOWLEDGE_PATH)

    chunks = paragraph_chunk_knowledge_base(
        documents,
        chunk_size=100
    )

    model = load_embedding_model()
    chunks = embed_chunks(chunks, model)

    client = create_vector_store(chunks)
    reranker = load_reranker()

    questions = load_questions(QUESTIONS_PATH)

    #for development
    #questions = questions[:5]

    evaluate_answers(
        questions,
        chunks,
        model,
        client,
        reranker
    )

    client.close()


if __name__ == "__main__":
    main()