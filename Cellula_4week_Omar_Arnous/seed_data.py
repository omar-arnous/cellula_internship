# seed_data.py
"""
Master seeding script to orchestrate the RAG pipeline:
1. Download Wikipedia docs
2. Embed the documents
3. Create ChromaDB and store embeddings
"""

import sys
import os

# Make sure src folder is in path if your scripts are inside src/
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

try:
    from src.download_docs import download_and_process_documents
    from src.embed_docs import embed_documents
    from src.create_chroma import create_chroma_db
except ImportError as e:
    print(f"Error importing modules: {e}")
    raise

def main():
    print("\n=== Step 1: Download & process Wikipedia documents ===")
    download_and_process_documents()

    print("\n=== Step 2: Embed the document chunks ===")
    embed_documents()

    print("\n=== Step 3: Create ChromaDB and store embeddings ===")
    create_chroma_db()

    print("\n✅ All steps completed successfully!")

if __name__ == "__main__":
    main()
