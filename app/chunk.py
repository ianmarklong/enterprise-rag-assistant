def chunk_document(document, chunk_size=100, overlap=20):
    words = document["content"].split() #gives list

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]

        chunk = {
            "content": " ".join(chunk_words),
            "source": document["source"],
            "category": document["category"],
            "chunk_id": chunk_id
        }

        chunks.append(chunk)

        start += chunk_size - overlap
        chunk_id += 1

    return chunks

def chunk_knowledge_base(documents, chunk_size=100, overlap=20):
    all_chunks = []

    for document in documents:
        chunks = chunk_document(document, chunk_size, overlap)
        all_chunks.extend(chunks)

    return all_chunks

