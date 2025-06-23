import streamlit as st
import tempfile
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
import time

# Load environment variables
load_dotenv()

# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"))

@st.cache_resource
def get_embedding_model():
    return OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    )

def get_qdrant_config():
    """Get Qdrant configuration from secrets or environment variables"""
    try:
        url = None
        api_key = None
        
        # Try to get from Streamlit secrets first
        if hasattr(st, 'secrets'):
            url = st.secrets.get("QDRANT_URL")
            api_key = st.secrets.get("QDRANT_API_KEY")
        
        # Fallback to environment variables
        if not url:
            url = os.getenv("QDRANT_URL")
        if not api_key:
            api_key = os.getenv("QDRANT_API_KEY")
            
        return {
            "url": url,
            "api_key": api_key
        }
    except Exception as e:
        st.error(f"Error getting Qdrant config: {str(e)}")
        return {
            "url": None,
            "api_key": None
        }

def store_vector_db(vector_store, collection_name):
    """Store vector database in session state with caching"""
    st.session_state.vector_store = vector_store
    st.session_state.collection_name = collection_name
    return vector_store

def process_pdf(uploaded_file):
    """Process uploaded PDF and create embeddings"""
    try:
        # Get Qdrant configuration first
        qdrant_config = get_qdrant_config()
        
        # Validate Qdrant configuration
        if not qdrant_config["url"] or not qdrant_config["api_key"]:
            st.error("❌ Qdrant configuration is missing. Please check your QDRANT_URL and QDRANT_API_KEY.")
            return None, None, 0
        
        # Create a temporary file to save the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load the PDF
        loader = PyPDFLoader(file_path=tmp_file_path)
        docs = loader.load()
        
        # Chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=400,
        )
        split_docs = text_splitter.split_documents(documents=docs)
        
        # Get embedding model
        embedding_model = get_embedding_model()
        
        # Create unique collection name based on file name and timestamp
        collection_name = f"pdf_{int(time.time())}"
        
        # Create vector store with Qdrant Cloud
        vector_store = QdrantVectorStore.from_documents(
            documents=split_docs,
            url=qdrant_config["url"],
            api_key=qdrant_config["api_key"],
            collection_name=collection_name,
            embedding=embedding_model
        )
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return vector_store, collection_name, len(split_docs)
    
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        # Add more detailed error information for debugging
        if "qdrant" in str(e).lower():
            st.error("Qdrant connection error. Please check your Qdrant Cloud configuration.")
        elif "openai" in str(e).lower():
            st.error("OpenAI API error. Please check your OpenAI API key.")
        
        # Clean up temporary file if it exists
        try:
            if 'tmp_file_path' in locals():
                os.unlink(tmp_file_path)
        except:
            pass
            
        return None, None, 0

def search_and_generate_response(query, vector_store):
    """Search vector database and generate response"""
    try:
        # Vector Similarity Search
        search_results = vector_store.similarity_search(query=query, k=4)
        
        # Create context from search results
        context = "\n\n\n".join([
            f"Page Content: {result.page_content}\n Page Number: {result.metadata.get('page', 'N/A')}\n File Location: {result.metadata.get('source', 'N/A')}"
            for result in search_results
        ])
        
        # System prompt
        system_prompt = f"""
        You are a helpful AI Assistant who answers user queries based on the available context
        retrieved from a PDF file along with page contents and page number.

        You should only answer the user based on the following context and navigate the user
        to open the right page number to know more.

        Context:
        {context}
        """
        
        # Get OpenAI client and generate response
        client = get_openai_client()
        chat_completion = client.chat.completions.create(
            model="gpt-4.1-mini",  # Fixed model name
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ]
        )
        
        return chat_completion.choices[0].message.content
    
    except Exception as e:
        return f"Error generating response: {str(e)}"

def main():
    st.set_page_config(
        page_title="PDF RAG Assistant",
        page_icon="📚",
        layout="wide"
    )
    
    # Check if required secrets are configured
    qdrant_config = get_qdrant_config()
    openai_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    if not all([qdrant_config["url"], qdrant_config["api_key"], openai_key]):
        st.error("❌ Missing required configuration. Please check your secrets configuration.")
        st.markdown("""
        **Required Secrets:**
        - `QDRANT_URL`: Your Qdrant Cloud cluster URL
        - `QDRANT_API_KEY`: Your Qdrant Cloud API key
        - `OPENAI_API_KEY`: Your OpenAI API key
        """)
        st.stop()
    
    # Initialize session state
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'collection_name' not in st.session_state:
        st.session_state.collection_name = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'show_chat' not in st.session_state:
        st.session_state.show_chat = False
    
    # Show chat interface if both conditions are met
    if st.session_state.pdf_processed and st.session_state.show_chat:
        # Header with reset option
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title("💬 Chat with Your PDF")
        with col2:
            if st.button("📎 Upload New PDF"):
                # Reset session state
                st.session_state.vector_store = None
                st.session_state.collection_name = None
                st.session_state.chat_history = []
                st.session_state.pdf_processed = False
                st.session_state.show_chat = False
                st.rerun()
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("💭 Conversation History")
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                with st.expander(f"Q{i+1}: {question[:50]}..." if len(question) > 50 else f"Q{i+1}: {question}"):
                    st.write("**Question:**", question)
                    st.write("**Answer:**", answer)
        
        # Chat input
        st.subheader("🤔 Ask a Question")
        
        # Create a form for better UX
        with st.form("chat_form", clear_on_submit=True):
            user_query = st.text_area(
                "What would you like to know about the document?",
                placeholder="e.g., What is the main topic of this document?",
                height=100
            )
            submitted = st.form_submit_button("🔍 Ask", type="primary")
        
        if submitted and user_query.strip():
            with st.spinner("Searching document and generating response..."):
                response = search_and_generate_response(user_query, st.session_state.vector_store)
                
                # Add to chat history
                st.session_state.chat_history.append((user_query, response))
                
                # Display current response
                st.success("Response generated!")
                with st.container():
                    st.write("**Your Question:**", user_query)
                    st.write("**Answer:**")
                    st.write(response)
        
        # Sidebar with information
        with st.sidebar:
            st.header("ℹ️ Session Info")
            st.write(f"**Collection:** {st.session_state.collection_name}")
            st.write(f"**Questions Asked:** {len(st.session_state.chat_history)}")
            
            st.header("💡 Tips")
            st.markdown("""
            - Ask specific questions about the content
            - Reference page numbers mentioned in responses
            - Try different phrasings if you don't get the answer you need
            - Questions work best when they relate to the document content
            """)
    
    # Show upload interface if not ready for chat
    else:
        st.title("📚 PDF RAG Assistant")
        st.header("Upload Your PDF Document")
        
        st.markdown("""
        Welcome to the PDF RAG Assistant! Upload a PDF document and I'll help you find information from it.
        
        **How it works:**
        1. Upload your PDF document
        2. I'll process and index the content
        3. Ask me questions about the document
        """)
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to get started"
        )
        
        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            # Display file details
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.2f} KB"
            }
            st.json(file_details)
            
            # Process button
            if st.button("🚀 Process PDF", type="primary"):
                with st.spinner("Processing PDF... This may take a few moments."):
                    vector_store, collection_name, num_chunks = process_pdf(uploaded_file)
                    
                    if vector_store is not None:
                        # Store the vector store securely
                        store_vector_db(vector_store, collection_name)
                        st.session_state.pdf_processed = True
                        
                        st.success(f"✅ PDF processed successfully!")
                        st.info(f"📄 Created {num_chunks} text chunks for search")
                        st.balloons()
        
        # If PDF is processed, show the Start Chatting button
        if st.session_state.pdf_processed and not st.session_state.show_chat:
            st.info("✨ Your PDF is ready! Click the button below to start chatting.")
            if st.button("💬 Start Chatting!", type="primary", key="start_chat_main"):
                st.session_state.show_chat = True
                st.rerun()

if __name__ == "__main__":
    main()