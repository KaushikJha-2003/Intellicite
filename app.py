import os
import shutil
import streamlit as st
import chromadb
import time
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader, YoutubeLoader, PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

st.set_page_config(page_title="Intellicite", layout="wide")

DATA_DIR = "data"
CHROMA_DIR = "chroma_store"
COLLECTION_NAME = f"intellicite_{int(time.time())}"
os.makedirs(DATA_DIR, exist_ok=True)

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! Models are loading..."}]
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "llm" not in st.session_state:
        st.session_state.llm = None
    if "embeddings_model" not in st.session_state:
        st.session_state.embeddings_model = None
    if "persona_select" not in st.session_state:
        st.session_state.persona_select = "Helpful Assistant"

initialize_session_state()

@st.cache_resource
def load_all_models():
    """Create and return all models using the Google Gemini API."""
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("Google API Key not found. Please create a .env file.")
        return None, None
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2", 
            model_kwargs={"device": "cpu"}
        )
        llm_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
        return embedding_model, llm_model
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None, None

def safe_cleanup_chroma():
    """Safely cleanup ChromaDB directory"""
    try:
        if os.path.exists(CHROMA_DIR):
            import gc
            gc.collect()
            
            for attempt in range(3):
                try:
                    shutil.rmtree(CHROMA_DIR)
                    break
                except PermissionError:
                    time.sleep(1)
                    continue
                except Exception:
                    break
        
        os.makedirs(CHROMA_DIR, exist_ok=True)
        return True
    except Exception as e:
        st.warning(f"Could not fully cleanup database: {e}")
        return False

def process_documents(docs, embedding_model):
    """Process documents and create vector store with improved error handling"""
    if not docs:
        st.error("No documents provided")
        return None
    
    if not embedding_model:
        st.error("Embedding model not available")
        return None
    
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        document_chunks = text_splitter.split_documents(docs)
        
        if not document_chunks:
            st.error("No text chunks created from documents")
            return None
        
        try:
            unique_collection = f"docs_{int(time.time())}_{len(document_chunks)}"
            
            vector_store = Chroma.from_documents(
                documents=document_chunks,
                embedding=embedding_model,
                collection_name=unique_collection
            )
            
            st.success(f"Created vector store with {len(document_chunks)} chunks")
            return vector_store
            
        except Exception as chroma_error:
            st.error(f"ChromaDB error: {chroma_error}")
            
            try:
                safe_cleanup_chroma()
                client = chromadb.PersistentClient(path=CHROMA_DIR)
                
                vector_store = Chroma.from_documents(
                    documents=document_chunks,
                    embedding=embedding_model,
                    client=client,
                    collection_name=f"fallback_{int(time.time())}"
                )
                return vector_store
                
            except Exception as fallback_error:
                st.error(f"Fallback also failed: {fallback_error}")
                return None
                
    except Exception as e:
        st.error(f"Document processing error: {e}")
        return None

def create_prompt_template(persona):
    """Create a prompt template for Gemini."""
    persona_instructions = {
        "Helpful Assistant": "You are a helpful and friendly AI assistant.",
        "Technical Expert": "You are a technical expert with deep knowledge in engineering and science.",
        "Business Analyst": "You are a sharp business analyst providing strategic insights.",
        "ELI5 (Explain Like I'm 5)": "You explain complex topics in very simple, easy-to-understand terms.",
    }
    instruction = persona_instructions.get(persona, "You are a helpful AI assistant.")
    
    template = f"""{instruction}
Answer the user's question based only on the context provided.
Use clear, structured markdown formatting.

CONTEXT:
{{context}}

QUESTION:
{{question}}

ANSWER:
"""
    return PromptTemplate(template=template, input_variables=["context", "question"])

def format_docs(docs):
    """Helper function to format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(vector_store, llm_model, persona):
    """Create a RAG chain using LangChain Expression Language (LCEL)."""
    try:
        if not vector_store:
            st.error("Vector store not available")
            return None
            
        if not llm_model:
            st.error("LLM model not available")
            return None
            
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        prompt = create_prompt_template(persona)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm_model
            | StrOutputParser()
        )
        return rag_chain
    except Exception as e:
        st.error(f"Error creating RAG chain: {e}")
        return None

if st.session_state.llm is None:
    with st.spinner("Loading AI models... Please wait."):
        st.session_state.embeddings_model, st.session_state.llm = load_all_models()
        if st.session_state.llm:
            st.session_state.messages = [{"role": "assistant", "content": "Models loaded! Please process a document."}]

with st.sidebar:
    st.header("Setup")
    
    if st.button("Reset Database"):
        st.session_state.vectorstore = None
        if safe_cleanup_chroma():
            st.success("Database reset successfully!")
        else:
            st.warning("Database partially reset - restart app if issues persist")
    
    if st.session_state.llm:
        st.success("Models loaded successfully!")
        st.header("AI Persona")
        personas = ["Helpful Assistant", "Technical Expert", "Business Analyst", "ELI5 (Explain Like I'm 5)"]
        st.session_state.persona_select = st.selectbox("Choose AI personality", personas)
        
        st.header("Document Source")
        source_option = st.radio("Select source type", ["PDF Upload", "Website", "YouTube"])
        
        if source_option == "PDF Upload":
            pdf_file = st.file_uploader("Choose PDF file", type="pdf")
            if st.button("Process PDF") and pdf_file:
                with st.spinner("Processing PDF..."):
                    try:
                        file_path = os.path.join(DATA_DIR, pdf_file.name)
                        with open(file_path, "wb") as f:
                            f.write(pdf_file.getvalue())
                        
                        loader = PyPDFLoader(file_path)
                        documents = loader.load()
                        
                        if documents:
                            st.session_state.vectorstore = process_documents(
                                documents, st.session_state.embeddings_model
                            )
                            if st.session_state.vectorstore:
                                st.success("PDF processed successfully!")
                                st.session_state.messages = [
                                    {"role": "assistant", "content": " PDF ready! Ask me anything about it."}
                                ]
                                st.rerun()
                            else:
                                st.error("Failed to create vector store from PDF")
                        else:
                            st.error("No content extracted from PDF")
                            
                    except Exception as e:
                        st.error(f"PDF processing failed: {e}")

        elif source_option == "Website":
            website_url = st.text_input("Enter website URL", placeholder="https://example.com")
            if st.button("Process Website") and website_url:
                with st.spinner("Processing website..."):
                    try:
                        if not website_url.startswith(('http://', 'https://')):
                            st.error("Please enter a valid URL starting with http:// or https://")
                        else:
                            loader = WebBaseLoader(website_url)
                            documents = loader.load()
                            
                            if documents:
                                st.session_state.vectorstore = process_documents(
                                    documents, st.session_state.embeddings_model
                                )
                                if st.session_state.vectorstore:
                                    st.success("Website processed successfully!")
                                    st.session_state.messages = [
                                        {"role": "assistant", "content": "Website ready! What would you like to know?"}
                                    ]
                                    st.rerun()
                                else:
                                    st.error("Failed to create vector store from website")
                            else:
                                st.error("No content extracted from website")
                                
                    except Exception as e:
                        st.error(f"Website processing failed: {e}")
                        
        elif source_option == "YouTube":
            youtube_url = st.text_input("Enter YouTube URL", placeholder="https://youtube.com/watch?v=...")
            if st.button("Process YouTube") and youtube_url:
                with st.spinner("Processing YouTube video..."):
                    try:
                        if "youtube.com" not in youtube_url and "youtu.be" not in youtube_url:
                            st.error("Please enter a valid YouTube URL")
                        else:
                            loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=True)
                            documents = loader.load()
                            
                            if documents:
                                st.session_state.vectorstore = process_documents(
                                    documents, st.session_state.embeddings_model
                                )
                                if st.session_state.vectorstore:
                                    st.success("YouTube video processed successfully!")
                                    st.session_state.messages = [
                                        {"role": "assistant", "content": "🎥 Video ready! Ask me about the content."}
                                    ]
                                    st.rerun()
                                else:
                                    st.error("Failed to create vector store from YouTube")
                            else:
                                st.error("No content extracted from YouTube video")
                                
                    except Exception as e:
                        st.error(f"YouTube processing failed: {e}")
    else:
        st.error("Models failed to load. Please check your .env file and Google API key.")

# ---- Main Chat Interface ----
st.title("Intellicite: AI Document Assistant 💡")
st.markdown("---")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_question = st.chat_input("Ask a question about your document...")

if user_question:
    if not st.session_state.llm:
        st.warning("Please ensure your Google API key is correct in the .env file.")
    elif not st.session_state.vectorstore:
        st.warning("Please process a document first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)
        
        with st.spinner("Generating response..."):
            try:
                rag_chain = create_rag_chain(
                    st.session_state.vectorstore, 
                    st.session_state.llm, 
                    st.session_state.persona_select
                )
                
                if rag_chain:
                    answer = rag_chain.invoke(user_question)
                    
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error("Failed to create RAG chain")
                    
            except Exception as e:
                error_msg = f"Error generating response: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

st.markdown("---")
st.markdown("*Powered by Google Gemini & LangChain* ")
