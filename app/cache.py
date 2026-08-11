import pickle
from pathlib import Path

def get_knowledge_fingerprint(folder_path): #fingerprint to check if knowledge base and cache are different
    folder_path = Path(folder_path)

    files = sorted(folder_path.rglob("*.md"))#searches for markdown files

    fingerprint = []

    for file in files:
        fingerprint.append( #relative path and last modified time
            (
                str(file.relative_to(folder_path)),
                file.stat().st_mtime
            )
        )
    return fingerprint

def save_chunk_cache(chunks, cache_path,fingerprint):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cache_data = {
        "fingerprint": fingerprint,
        "chunks": chunks
    }

    with open(cache_path, "wb") as file:
        pickle.dump(cache_data, file) #serialises python object into files


def load_chunk_cache(cache_path,current_fingerprint):
    cache_path = Path(cache_path)

    if not cache_path.exists():
        return None

    with open(cache_path, "rb") as file:
        cache_data = pickle.load(file) #recontruct python object

    if cache_data['fingerprint'] != current_fingerprint: #if fingerprint on cached data is different
        return None #none will make main.py run save_chunk_cache()

    return cache_data["chunks"]