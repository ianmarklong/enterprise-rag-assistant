#__init__.py makes python treat the folder as a package
from pathlib import Path

from app.chunk import paragraph_chunk_knowledge_base
from app.embed import load_embedding_model, embed_chunks
from app.load import load_knowledge_base


project_root = Path(__file__).resolve().parents[1]
knowledge_path = project_root / "knowledge_base"

from app.cache import (
    get_knowledge_fingerprint,
    load_chunk_cache,
    save_chunk_cache
)
cache_path = project_root / "data" / "chunk_cache.pkl"

fingerprint = get_knowledge_fingerprint(knowledge_path)

chunks = load_chunk_cache(
    cache_path,
    fingerprint
) #load chunk cache if it exists

model = load_embedding_model() #always needed since we need to embed user's query

if chunks is None:
    print("No cache found. Building embeddings...")
    #either because doesnt exist or fingerprint mismatch
    documents = load_knowledge_base(knowledge_path)

    chunks = paragraph_chunk_knowledge_base( #chunk
        documents,
        chunk_size=100
    )

    chunks = embed_chunks(chunks, model) #embed

    save_chunk_cache(
        chunks,
        cache_path,
        fingerprint
    ) #cache

    print("Embedding cache saved.")

else:
    print("Loaded chunks and embeddings from cache.")

from app.vector_store import (
    create_vector_store,
    search_vector_store
)

client = create_vector_store(chunks)

from app.retrieve import retrieve

question = "How do I get access to GPU resources?"

#would loop through every chunk and calculate cosine similarity
'''
results = retrieve(
    question,
    chunks,
    model,
    top_k=3
)
'''

query_embedding = model.encode_query(
    question,
    convert_to_numpy=True
)

results = search_vector_store(
    client,
    query_embedding,
    top_k=3,
    category ='infrastructure'
)
client.close()
print(f"\nQuestion: {question}\n")

for result in results:
    print(f"Source: {result['source']}")
    print(f"Category: {result['category']}")
    print(f"Score: {result['score']:.4f}")
    print(result["content"])
    print("-" * 50)


from app.prompt import build_prompt

prompt = build_prompt(question, results)

print(prompt)

from app.generate import generate_answer

#answer = generate_answer(prompt) #dont run because i am not paying for an openai api key for this

print("\nANSWER")
print("-" * 50)
#print(answer)

#cd "C:\Users\ianma\Enterprise RAG Assistant" 
#.venv\Scripts\Activate.ps1  
#$env:OPENAI_API_KEY="sk-qrstefghuvwxabcdqrstefghuvwxabcdqrstefgh"
#python -m app.main in enterprise rag assistant dir
