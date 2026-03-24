import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

load_dotenv()

client = QdrantClient(url="http://localhost:6333")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
COLLECTION_NAME = "company_data"

def ingest_document(file_path, allowed_roles):
    # 1. Load and Split
    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split()
    
    for doc in docs:
        doc.metadata["allowed_roles"] = allowed_roles

    # 2. Ensure Collection exists (Manual check to avoid the 'init_from' error)
    collections = client.get_collections().collections
    exists = any(c.name == COLLECTION_NAME for c in collections)
    
    if not exists:
        print(f"Creating collection: {COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

    # 3. Upload documents using the simpler 'from_documents' method
    # We pass the client directly to avoid internal re-initialization
    Qdrant.from_documents(
        docs,
        embeddings,
        url="http://localhost:6333",
        collection_name=COLLECTION_NAME,
    )
    print(f"Successfully ingested {file_path}")

if __name__ == "__main__":
    # Make sure these files exist in your data/ folder!
    ingest_document("data/Q3_Financials.pdf", allowed_roles=["finance", "c-level"])
    ingest_document("data/Employee_Payroll.pdf", allowed_roles=["hr", "c-level"])
    ingest_document("data/Company_Holidays.pdf", allowed_roles=["finance", "hr", "c-level"])