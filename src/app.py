import streamlit as st
import time
from chain import get_rag_chain
from guardrails import redact_pii, is_out_of_scope
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load API Keys from .env
load_dotenv()

# --- Page Configuration ---
st.set_page_config(page_title="Corporate Secure AI", page_icon="🏢", layout="wide")

st.title("🏢 Internal Company Chatbot")
st.markdown("---")

# --- Sidebar: Identity & Access Management (Mock RBAC) ---
with st.sidebar:
    st.header("🔐 User Identity")
    user_role = st.selectbox(
        "Select your Department (RBAC Testing):",
        ["finance", "hr", "marketing", "c-level"],
        index=0
    )
    
    st.info(f"Current Access: **{user_role.upper()}**")
    
    with st.expander("Settings & Monitoring"):
        st.write(f"Connected to Qdrant: ✅")
        st.write(f"LangSmith Tracing: ✅")
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display Chat Messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Logic ---
if prompt := st.chat_input("Ask a question about company data..."):
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        
        # --- Guardrail 1: Out of Scope Check ---
        status_placeholder.status("🛡️ Checking scope...")
        llm_routing = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        if is_out_of_scope(prompt, llm_routing):
            response_text = "⚠️ **Policy Violation:** I am restricted to answering questions regarding internal company data (HR, Finance, Marketing). Please contact IT for general queries."
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            status_placeholder.empty()
        
        else:
            # --- Guardrail 2: PII Redaction ---
            status_placeholder.status("🔍 Scanning for PII...")
            safe_query = redact_pii(prompt)
            if safe_query != prompt:
                st.warning(f"Sensitive info detected! Query was anonymized for safety.")

            # --- RAG Execution ---
            status_placeholder.status(f"📚 Searching {user_role} records...")
            try:
                # Initialize the chain with the specific user role
                rag_chain = get_rag_chain(user_role)
                
                # Run the chain
                response = rag_chain.invoke({"input": safe_query})
                answer = response["answer"]
                sources = response.get("context", [])

                # 4. Display Final Answer
                st.markdown(answer)
                
                # 5. Display Sources (Expandable)
                if sources:
                    with st.expander("View Trusted Sources"):
                        for i, doc in enumerate(sources):
                            st.caption(f"Source {i+1} (Role: {doc.metadata.get('allowed_roles')}):")
                            st.write(doc.page_content[:200] + "...")

                st.session_state.messages.append({"role": "assistant", "content": answer})
                status_placeholder.empty()

            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
                status_placeholder.empty()