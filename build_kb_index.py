"""
build_kb_index.py

Builds a ChromaDB knowledge-base index for the ISDO (Intelligent Service Desk
Orchestrator) project.

What it does:
    1. Reads all .md files from data/kb/
    2. Splits each article into chunks at "## " headings
    3. Stores all chunks in a ChromaDB collection called 'isdo_kb'
       (using Chroma's built-in default embedding function - no extra
       packages beyond `chromadb` are required)
    4. Runs 4 sample queries and prints the best-matching article name
       and a confidence score for each

Usage:
    pip install chromadb
    python build_kb_index.py
"""

import os
import re
import glob
import chromadb

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
KB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "kb")
CHROMA_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
COLLECTION_NAME = "isdo_kb"

SAMPLE_QUERIES = [
    "My VPN stopped working after I changed my password",
    "Outlook on my phone is not syncing new emails",
    "SAP login is failing for a lot of people at once",
    "I'm locked out of my account and need a password reset",
]


# ---------------------------------------------------------------------------
# Step 1 & 2: Read markdown files and split into chunks at "## " headings
# ---------------------------------------------------------------------------
def load_and_chunk_articles(kb_dir):
    """
    Reads every .md file in kb_dir and splits it into chunks on lines that
    start with a level-2 markdown heading ("## Heading"). Content before the
    first "## " heading (e.g. the title and metadata block) is kept as its
    own chunk. Level-3 headings ("### Step 1") stay inside their parent
    chunk since the split pattern only matches exactly "## ".

    Returns a list of dicts: {id, article, heading, text}
    """
    chunks = []
    md_files = sorted(glob.glob(os.path.join(kb_dir, "*.md")))

    if not md_files:
        raise FileNotFoundError(f"No .md files found in {kb_dir}")

    heading_pattern = re.compile(r"^##\s+(.*)$", re.MULTILINE)

    for filepath in md_files:
        article_name = os.path.basename(filepath)

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Find every "## " heading and its start position
        matches = list(heading_pattern.finditer(content))

        if not matches:
            # No "##" headings at all -- treat the whole file as one chunk
            text = content.strip()
            if text:
                chunks.append({
                    "id": f"{article_name}::chunk-0",
                    "article": article_name,
                    "heading": "(full document)",
                    "text": text,
                })
            continue

        # Preamble: everything before the first "## " heading (title, tags, etc.)
        preamble = content[: matches[0].start()].strip()
        if preamble:
            chunks.append({
                "id": f"{article_name}::chunk-0",
                "article": article_name,
                "heading": "(preamble)",
                "text": preamble,
            })

        # One chunk per "## " heading, spanning to the next "## " heading
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            heading_title = match.group(1).strip()
            chunk_text = content[start:end].strip()

            chunks.append({
                "id": f"{article_name}::chunk-{i + 1}",
                "article": article_name,
                "heading": heading_title,
                "text": chunk_text,
            })

    return chunks


# ---------------------------------------------------------------------------
# Step 3: Store chunks in a ChromaDB collection
# ---------------------------------------------------------------------------
def build_collection(chunks):
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    # Start fresh each run so re-running the script doesn't duplicate chunks
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"article": c["article"], "heading": c["heading"]} for c in chunks],
    )

    return collection


# ---------------------------------------------------------------------------
# Step 4: Run sample queries and report the best-matching article
# ---------------------------------------------------------------------------
def run_sample_queries(collection, queries):
    print("\n" + "=" * 70)
    print("SAMPLE QUERY RESULTS")
    print("=" * 70)

    for query in queries:
        results = collection.query(query_texts=[query], n_results=1)

        best_article = results["metadatas"][0][0]["article"]
        best_heading = results["metadatas"][0][0]["heading"]
        distance = results["distances"][0][0]

        # Chroma's default collection space is squared L2 ("hnsw:space": "l2"),
        # and the default embedding function returns L2-normalized vectors.
        # For unit vectors, squared-L2 distance = 2 - 2*cos_sim, so
        # cos_sim = 1 - distance/2. We report that cosine similarity as a
        # 0-100% "confidence" score, since it's easier to read than a raw
        # distance value.
        confidence_pct = max(0.0, (1 - distance / 2)) * 100

        print(f"\nQuery: \"{query}\"")
        print(f"  Best match : {best_article}  (section: {best_heading})")
        print(f"  Confidence : {confidence_pct:.1f}%  (distance={distance:.4f})")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Reading markdown articles from: {KB_DIR}")
    chunks = load_and_chunk_articles(KB_DIR)
    print(f"Loaded {len(chunks)} chunks from "
          f"{len(set(c['article'] for c in chunks))} articles.")

    print(f"\nBuilding ChromaDB collection '{COLLECTION_NAME}' at: {CHROMA_DB_DIR}")
    collection = build_collection(chunks)
    print(f"Stored {collection.count()} chunks in collection '{COLLECTION_NAME}'.")

    run_sample_queries(collection, SAMPLE_QUERIES)


if __name__ == "__main__":
    main()
