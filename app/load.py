from pathlib import Path

def load_document(path):
    path = Path(path)

    with open(path, "r", encoding="utf-8") as file:
        content = file.read()

    return {
        "content": content,
        "source": path.name,
        "category": path.parent.name
    }

def load_knowledge_base(folder_path):
    documents = []

    for file_path in Path(folder_path).rglob("*.md"):
        document = load_document(file_path)
        documents.append(document)

    return documents

