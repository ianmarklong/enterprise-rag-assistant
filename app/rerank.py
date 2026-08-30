import os

from sentence_transformers import CrossEncoder


RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def load_reranker():
    return CrossEncoder(RERANKER_MODEL)

def rerank(question, results, reranker, top_k=3):
    pairs = []

    for result in results:
        pairs.append([
            question,
            result["content"]
        ])

    scores = reranker.predict(pairs)

    #pair
    result_scores = zip(results,scores)

    #store 
    for result,score in result_scores:
        result['rerank_score'] = float(score)

    #sort
    results.sort(key = lambda result: result['rerank_score'], reverse = True)

    return results[:top_k]
