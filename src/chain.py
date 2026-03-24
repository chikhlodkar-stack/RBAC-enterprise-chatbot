import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Load credentials
llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize Client
client = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "company_data"

def get_rag_chain(user_role: str):
    # 1. Access the Vector Store via the Community class
    vectorstore = Qdrant(
        client=client, 
        collection_name=COLLECTION_NAME, 
        embeddings=embeddings
    )

    # 2. RBAC Filter Logic
    search_kwargs = {"k": 3} # Retrieve top 3 documents
    
    # C-level bypasses filters, everyone else is filtered
    if user_role.lower() != "c-level":
        search_kwargs["filter"] = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.allowed_roles",
                    match=models.MatchValue(value=user_role.lower())
                )
            ]
        )

    # 3. Setup Retriever
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    # 4. Standard RAG Prompt
    prompt = ChatPromptTemplate.from_template("""
    You are a professional corporate assistant. Use the following context to answer the question.
    If the context is empty or you don't have access, politely explain that you cannot find that information.
    
    Context: {context}
    Question: {input}
    Answer:""")

    # 5. Build Chains
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    return retrieval_chain