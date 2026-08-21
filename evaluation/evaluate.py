import json
from pathlib import Path

from app.load import load_knowledge_base
from app.chunk import paragraph_chunk_knowledge_base
from app.embed import load_embedding_model, embed_chunks
from app.rerank import load_reranker, rerank
from app.vector_store import create_vector_store, search_vector_store


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_PATH = PROJECT_ROOT / "knowledge_base"
QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "questions.json"


def load_questions(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def has_source_hit(results, expected_sources):
    # Return True when at least one result source
    # appears in expected_sources.
    for result in results:
        if result['source'] in expected_sources:
            return True
        
    return False


def evaluate_retrieval(questions, chunks, model, client, reranker):
    hit_at_1 = 0
    hit_at_3 = 0
    total_coverage_at_3 = 0
    total_reciprocal_rank = 0

    for item in questions:
        question = item["question"]
        expected_sources = item["expected_sources"]

        query_embedding = model.encode(
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

        rr = reciprocal_rank(
            results,
            expected_sources
        )

        total_reciprocal_rank += rr
        print(f"Reciprocal rank: {rr:.2f}")

        top_1_hit = has_source_hit(
            results[:1],
            expected_sources
        )

        top_3_hit = has_source_hit(
            results,
            expected_sources
        )

        if top_1_hit:
            hit_at_1 += 1

        if top_3_hit:
            hit_at_3 += 1

        coverage = expected_source_coverage(
            results,
            expected_sources
        )

        total_coverage_at_3 += coverage

        print(f"\nQuestion: {question}")
        print(f"Expected: {expected_sources}")

        # Add retrieved ranks here
        print("Retrieved:")

        for rank, result in enumerate(results, start=1):
            print(
                f"\n  {rank}. {result['source']}"
                f" — vector: {result['score']:.4f}"
                f" — rerank: {result['rerank_score']:.4f}"
            )

            print("  Start:", repr(result["content"][:120]))
            print("  End:  ", repr(result["content"][-120:]))

        print(f"Hit@1: {top_1_hit}")
        print(f"Hit@3: {top_3_hit}")
        print(f"Coverage@3: {coverage:.2%}")

    question_count = len(questions)

    print("\nRESULTS")
    print("-" * 50)
    print(f"Questions: {question_count}")
    print(f"Hit@1: {hit_at_1 / question_count:.2%}")
    print(f"Hit@3: {hit_at_3 / question_count:.2%}")
    print(
    f"MRR: "
    f"{total_reciprocal_rank / question_count:.3f}"
)

    # Add final coverage print here
    print(
        f"Expected-source coverage@3: "
        f"{total_coverage_at_3 / question_count:.2%}"
    )

def expected_source_coverage(results, expected_sources):
    if len(expected_sources) == 0:
        return 0.0

    retrieved_sources = {
        result["source"]
        for result in results
    }

    matched_sources = retrieved_sources.intersection(expected_sources)

    return len(matched_sources) / len(expected_sources)

def reciprocal_rank(results, expected_sources):
    for rank, result in enumerate(results, start=1):
        if result["source"] in expected_sources:
            return 1 / rank

    return 0

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

    evaluate_retrieval(
        questions,
        chunks,
        model,
        client,
        reranker
    )

    client.close()


if __name__ == "__main__":
    main()