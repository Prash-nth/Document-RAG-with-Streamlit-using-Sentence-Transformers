import os
import tempfile
from datetime import datetime
from typing import List

import streamlit as st
import bs4
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings


class SentenceTransformerEmbedder(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode([text], show_progress_bar=False)[0]
        return embedding.tolist()


# Streamlit App Initialization
st.title("🤔 Document RAG with Sentence Transformers")

# Session State Initialization
if 'processed_documents' not in st.session_state:
    st.session_state.processed_documents = []
if 'history' not in st.session_state:
    st.session_state.history = []
if 'similarity_threshold' not in st.session_state:
    st.session_state.similarity_threshold = 0.5  # Lowered for better recall
if 'document_store' not in st.session_state:
    st.session_state.document_store = []  # Store documents and embeddings
if 'embedder' not in st.session_state:
    st.session_state.embedder = SentenceTransformerEmbedder()


# Sidebar Configuration
st.sidebar.header("🎯 Search Configuration")
st.session_state.similarity_threshold = st.sidebar.slider(
    "Document Similarity Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Lower values return more documents but may include less relevant ones. Higher values are stricter."
)

# Clear Chat Button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.history = []
    st.session_state.document_store = []
    st.session_state.embedder = SentenceTransformerEmbedder()  # Reset embedder
    st.session_state.processed_documents = []
    st.rerun()


# Document Processing Functions
def process_pdf(file) -> List:
    """Process PDF file and add source metadata."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.getvalue())
            loader = PyPDFLoader(tmp_file.name)
            documents = loader.load()
            
            # Add source metadata
            for doc in documents:
                doc.metadata.update({
                    "source_type": "pdf",
                    "file_name": file.name,
                    "timestamp": datetime.now().isoformat()
                })
                
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            return text_splitter.split_documents(documents)
    except Exception as e:
        st.error(f"📄 PDF processing error: {str(e)}")
        return []


def process_web(url: str) -> List:
    """Process web URL and add source metadata."""
    try:
        loader = WebBaseLoader(
            web_paths=(url,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header", "content", "main")
                )
            )
        )
        documents = loader.load()
        
        # Add source metadata
        for doc in documents:
            doc.metadata.update({
                "source_type": "url",
                "url": url,
                "timestamp": datetime.now().isoformat()
            })
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return text_splitter.split_documents(documents)
    except Exception as e:
        st.error(f"🌐 Web processing error: {str(e)}")
        return []


def store_documents(documents):
    """Store documents and their embeddings in memory."""
    try:
        texts = [doc.page_content for doc in documents]
        embeddings = st.session_state.embedder.embed_documents(texts)
        
        # Store documents and embeddings
        for doc, embedding in zip(documents, embeddings):
            st.session_state.document_store.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "embedding": embedding
            })
        st.success("✅ Documents stored successfully!")
    except Exception as e:
        st.error(f"🔴 Document storage error: {str(e)}")


def retrieve_documents(query: str, threshold: float = 0.5, k: int = 3) -> List:
    """Retrieve relevant documents using sklearn cosine similarity."""
    if not st.session_state.document_store:
        return []
    
    try:
        query_embedding = np.array(st.session_state.embedder.embed_query(query)).reshape(1, -1)
        relevant_docs = []
        
        # Compute cosine similarity for each document
        for doc in st.session_state.document_store:
            doc_embedding = np.array(doc["embedding"]).reshape(1, -1)
            similarity = cosine_similarity(query_embedding, doc_embedding)[0][0]
            if similarity > threshold:
                relevant_docs.append({
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "score": similarity
                })
        
        # Sort by similarity and limit to k results
        relevant_docs = sorted(relevant_docs, key=lambda x: x["score"], reverse=True)[:k]
        return relevant_docs
    except Exception as e:
        st.error(f"❌ Retrieval error: {str(e)}")
        return []


def generate_response(query: str, docs: List) -> str:
    """Generate a concise response based on retrieved documents."""
    if not docs:
        return "No relevant information found in the documents."
    
    # Extract the most relevant document's content
    most_relevant = max(docs, key=lambda x: x["score"])
    content = most_relevant["content"]
    
    # Simple response: extract first few sentences or relevant snippet
    sentences = content.split(". ")[:3]  # Take first 3 sentences
    response = ". ".join(sentences) + (". " if sentences else "")
    
    # Add source information
    source_type = most_relevant["metadata"].get("source_type", "unknown")
    source_name = most_relevant["metadata"].get("file_name" if source_type == "pdf" else "url", "unknown")
    response += f"\n\nSource: {source_name} (Similarity: {most_relevant['score']:.2f})"
    
    return response


# Main Application Flow
# File/URL Upload Section
st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
web_url = st.sidebar.text_input("Or enter URL")

# Process documents
if uploaded_file:
    file_name = uploaded_file.name
    if file_name not in st.session_state.processed_documents:
        with st.spinner('Processing PDF...'):
            texts = process_pdf(uploaded_file)
            if texts:
                store_documents(texts)
                st.session_state.processed_documents.append(file_name)
                st.success(f"✅ Added PDF: {file_name}")

if web_url:
    if web_url not in st.session_state.processed_documents:
        with st.spinner('Processing URL...'):
            texts = process_web(web_url)
            if texts:
                store_documents(texts)
                st.session_state.processed_documents.append(web_url)
                st.success(f"✅ Added URL: {web_url}")

# Display sources in sidebar
if st.session_state.processed_documents:
    st.sidebar.header("📚 Processed Sources")
    for source in st.session_state.processed_documents:
        if source.endswith('.pdf'):
            st.sidebar.text(f"📄 {source}")
        else:
            st.sidebar.text(f"🌐 {source}")

# Chat Interface
prompt = st.chat_input("Ask about your documents...")

if prompt:
    # Add user message to history
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Retrieve relevant documents
    with st.spinner("🔍 Retrieving documents..."):
        try:
            docs = retrieve_documents(
                query=prompt,
                threshold=st.session_state.similarity_threshold,
                k=3  # Reduced for more focused results
            )
            
            if docs:
                st.info(f"📊 Found {len(docs)} relevant documents (similarity > {st.session_state.similarity_threshold})")
                response = generate_response(prompt, docs)
            else:
                st.info("ℹ️ No relevant documents found.")
                response = "No relevant information found in the documents."

            # Add assistant response to history
            st.session_state.history.append({
                "role": "assistant",
                "content": response
            })
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.write(response)
                
                # Show sources if available
                if docs:
                    with st.expander("🔍 See document sources"):
                        for i, doc in enumerate(docs, 1):
                            source_type = doc["metadata"].get("source_type", "unknown")
                            source_icon = "📄" if source_type == "pdf" else "🌐"
                            source_name = doc["metadata"].get("file_name" if source_type == "pdf" else "url", "unknown")
                            st.write(f"{source_icon} Source {i} from {source_name} (similarity: {doc['score']:.2f}):")
                            st.write(f"{doc['content'][:200]}...")

        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")