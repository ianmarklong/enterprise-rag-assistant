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

def paragraph_chunk_document(document, chunk_size=100):
    paragraphs = [
        paragraph.strip()
        for paragraph in document["content"].split("\n\n")
        if paragraph.strip()
    ]

    chunks = []
    current_paragraphs = []
    current_word_count = 0
    chunk_id = 0

    for paragraph in paragraphs:
        paragraph_word_count = len(paragraph.split())

        if (
            current_paragraphs
            and current_word_count + paragraph_word_count > chunk_size
        ):
            chunks.append({
                "content": "\n\n".join(current_paragraphs),
                "source": document["source"],
                "category": document["category"],
                "chunk_id": chunk_id
            })

            chunk_id += 1
            current_paragraphs = []
            current_word_count = 0

        current_paragraphs.append(paragraph)
        current_word_count += paragraph_word_count

    if current_paragraphs:
        chunks.append({
            "content": "\n\n".join(current_paragraphs),
            "source": document["source"],
            "category": document["category"],
            "chunk_id": chunk_id
        })

    return chunks

def paragraph_chunk_knowledge_base(documents, chunk_size=100):
    all_chunks = []

    for document in documents:
        chunks = paragraph_chunk_document(
            document,
            chunk_size
        )

        all_chunks.extend(chunks)

    return all_chunks