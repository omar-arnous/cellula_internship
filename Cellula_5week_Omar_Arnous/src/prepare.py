from load import subset
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings


# 1. Creating the embedding function
langchain_embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Initialize in-memory Chroma 
vectordb = Chroma(
    collection_name="human_eval_demo",
    embedding_function=langchain_embedder,
    persist_directory="../data/vectorstores/human_eval_demo"
)


# 3. Creating all the item components ( id, embedding, metadata, content )
docs_to_add = []
for index, row in subset.iterrows():
    page_content = row['prompt'] + " " + row['canonical_solution']
    # Use task_id as metadata, and also include prompt and canonical_solution
    metadata = {
        "task_id": row['task_id'],
        "prompt": row['prompt'],
        "canonical_solution": row['canonical_solution']
    }
    docs_to_add.append(Document(page_content=page_content, metadata=metadata))


# 4. Add Document objects to the vector store
vectordb.add_documents(docs_to_add)
print(f"Added {len(docs_to_add)} documents to Chroma collection.")

# # Quick test query
# query = "Write a function that sums two numbers"
# results = vectordb.similarity_search(query, k=1)
# top_doc = results[0]
# print("Page content:\n", top_doc.page_content)
# print("\nMetadata:\n", top_doc.metadata)