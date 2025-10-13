import os
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import NLTKTextSplitter
import nltk
import json

nltk.download('punkt')
nltk.download("punkt_tab")  

topics = [
    "Customer support best practices",
    "Customer service communication skills",
    "Customer relationship management",
    "Customer satisfaction",
    "Customer feedback",
    "Customer experience",
    "Technical support",
    "Help desk",
    "Complaint handling",
    "Call center operations",
    "Email support",
    "Live chat support",
    "Support ticket systems",
    "Knowledge base management",
    "Customer retention",
    "Customer empathy",
    "Product return policy",
    "Refund process",
    "Service-level agreement",
    "Troubleshooting guide"
]

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
os.makedirs(DATA_DIR, exist_ok=True)

def download_and_process_documents():
    docs = []
    for topic in topics:
        print(f"Loading docs for: {topic}")
        loader = WikipediaLoader(query=topic, load_max_docs=3, doc_content_chars_max=40_000)
        new_docs = loader.load()
        print(f"  -> Loaded {len(new_docs)} docs.")
        docs.extend(new_docs)

    print(f"Loaded {len(docs)} documents.")

    documents = [ doc.page_content for doc in docs ]
    metadata = [ {"title":doc.metadata["title"], "source":doc.metadata["source"]} for doc in docs ]

    raw_path = os.path.join(DATA_DIR, "raw_wikipedia_docs.json")

    # Save documents list
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump([{"content": content, "metadata": meta} for content, meta in zip(documents, metadata)], f, ensure_ascii=False, indent=4)

    print("Saved raw documents with metadata.")

    # Split documents
    text_splitter = NLTKTextSplitter(chunk_size=800, chunk_overlap=100)
    split_docs = text_splitter.create_documents(documents, metadatas=metadata)
    print(f"Split into {len(split_docs)} chunks.")

    # Save chunks and metadata to json file
    chunks_path = os.path.join(DATA_DIR, "wikipedia_docs_chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump([{"content": doc.page_content, "metadata": doc.metadata} for doc in split_docs], f, ensure_ascii=False, indent=4)

    print("Chunks saved to wikipedia_docs_chunks.json")

if __name__ == "__main__":
    download_and_process_documents()