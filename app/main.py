#__init__.py makes python treat the folder as a package
from pathlib import Path

from app.load import load_knowledge_base
from app.chunk import chunk_knowledge_base


project_root = Path(__file__).resolve().parents[1]
knowledge_path = project_root / "knowledge_base"

documents = load_knowledge_base(knowledge_path)
chunks = chunk_knowledge_base(documents)

print(f"Documents: {len(documents)}")
print(f"Chunks: {len(chunks)}")

print(chunks[0])

from app.embed import load_embedding_model, embed_chunks

model = load_embedding_model()
chunks = embed_chunks(chunks, model)

print(chunks[0]["embedding"].shape)

from app.retrieve import retrieve

question = "How do I reset my password?"

results = retrieve(
    question,
    chunks,
    model,
    top_k=3
)

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
