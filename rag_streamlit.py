"""
Retrieval-Augmented Generation (RAG) System with Streamlit Interface

This application allows users to upload PDF files and ask questions based on their contents.
It uses LangChain for retrieval and QA, FAISS for vector storage, and OpenAI/SentenceTransformers for embeddings.
"""

import os
import hashlib
import pickle
from pathlib import Path
from typing import List, Optional

# Optional: Proxy support via environment variables
# If you need proxy support, set these before running:
# os.environ["HTTP_PROXY"] = "http://127.0.0.1:8080"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:8080"
# Or set them in your system environment variables

import streamlit as st

# IMPORTANT: st.set_page_config must be the FIRST Streamlit command
# This must be called before any other Streamlit commands (st.error, st.stop, etc.)
try:
    st.set_page_config(
        page_title="RAG PDF Q&A System",
        page_icon="📚",
        layout="wide"
    )
except Exception:
    # If page config already set, continue
    pass

# Text splitter - try newer import first, fallback to older
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        # Fallback for older versions
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        st.error("❌ Could not import RecursiveCharacterTextSplitter. Please install langchain-text-splitters.")
        st.stop()

# Vector store - with error handling
try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    st.error("❌ Could not import FAISS. Please install langchain-community and faiss-cpu.")
    st.stop()

# Chat models - with error handling
try:
    from langchain_community.chat_models import ChatOpenAI
    OPENAI_LLM_AVAILABLE = True
except ImportError:
    OPENAI_LLM_AVAILABLE = False

# Google Gemini - try to import
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    try:
        # Fallback to langchain_community if langchain_google_genai not available
        from langchain_community.chat_models import ChatGoogleGenerativeAI
        GEMINI_AVAILABLE = True
    except ImportError:
        GEMINI_AVAILABLE = False

# Note: RetrievalQA is deprecated in LangChain 1.0+
# We'll use LCEL (LangChain Expression Language) approach instead

# Document - try newer import first, fallback to older
try:
    from langchain_core.documents import Document
except ImportError:
    try:
        # Fallback for older versions
        from langchain.schema import Document
    except ImportError:
        st.error("❌ Could not import Document. Please install langchain-core or langchain.")
        st.stop()

# PDF processing libraries
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

# Embeddings
# Use langchain_openai for OpenAIEmbeddings (recommended, newer package)
try:
    from langchain_openai import OpenAIEmbeddings
    OPENAI_EMBEDDINGS_AVAILABLE = True
except ImportError:
    # Fallback to langchain_community if langchain_openai not available
    try:
        from langchain_community.embeddings import OpenAIEmbeddings
        OPENAI_EMBEDDINGS_AVAILABLE = True
    except ImportError:
        OPENAI_EMBEDDINGS_AVAILABLE = False

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5
CACHE_DIR = Path(".rag_cache")
EMBEDDINGS_CACHE_FILE = CACHE_DIR / "embeddings_cache.pkl"

# Initialize cache directory (with error handling for Streamlit Cloud)
try:
    CACHE_DIR.mkdir(exist_ok=True)
except Exception:
    # If cache directory can't be created, continue anyway
    pass


def get_file_hash(file_bytes: bytes) -> str:
    """Generate a hash for the file to use as cache key."""
    return hashlib.md5(file_bytes).hexdigest()


def extract_text_from_pdf_pymupdf(file_bytes: bytes, filename: str) -> List[Document]:
    """
    Extract text from PDF using PyMuPDF (fitz).
    Returns list of Document objects with page numbers in metadata.
    """
    documents = []
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            documents.append(Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "page": page_num,
                    "total_pages": len(doc)
                }
            ))
    
    doc.close()
    return documents


def extract_text_from_pdf_pdfplumber(file_bytes: bytes, filename: str) -> List[Document]:
    """
    Extract text from PDF using pdfplumber.
    Returns list of Document objects with page numbers in metadata.
    """
    documents = []
    
    import io
    pdf_file = io.BytesIO(file_bytes)
    
    with pdfplumber.open(pdf_file) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "page": page_num,
                        "total_pages": len(pdf.pages)
                    }
                ))
    
    return documents


def extract_text_from_pdf_pypdf2(file_bytes: bytes, filename: str) -> List[Document]:
    """
    Extract text from PDF using PyPDF2 (fallback).
    Returns list of Document objects with page numbers in metadata.
    """
    documents = []
    
    import io
    pdf_file = io.BytesIO(file_bytes)
    reader = PdfReader(pdf_file)
    
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            documents.append(Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "page": page_num,
                    "total_pages": len(reader.pages)
                }
            ))
    
    return documents


def extract_text_from_pdf(file_bytes: bytes, filename: str) -> List[Document]:
    """
    Extract text from PDF file using available library (priority: PyMuPDF > pdfplumber > PyPDF2).
    Returns list of Document objects with page numbers.
    """
    if PYMUPDF_AVAILABLE:
        try:
            return extract_text_from_pdf_pymupdf(file_bytes, filename)
        except Exception as e:
            st.warning(f"PyMuPDF extraction failed: {e}. Trying fallback...")
    
    if PDFPLUMBER_AVAILABLE:
        try:
            return extract_text_from_pdf_pdfplumber(file_bytes, filename)
        except Exception as e:
            st.warning(f"pdfplumber extraction failed: {e}. Trying fallback...")
    
    if PYPDF2_AVAILABLE:
        try:
            return extract_text_from_pdf_pypdf2(file_bytes, filename)
        except Exception as e:
            st.error(f"All PDF extraction methods failed: {e}")
            return []
    
    st.error("No PDF extraction library available. Please install PyMuPDF, pdfplumber, or PyPDF2.")
    return []


def get_embeddings():
    """
    Get embeddings model - OpenAI if API key available, otherwise SentenceTransformers.
    
    Note: Proxy support (if needed) via environment variables:
    - HTTP_PROXY=http://your.proxy:port
    - HTTPS_PROXY=http://your.proxy:port
    Do NOT use proxies= parameter (not supported in new OpenAI SDK).
    """
    # Try OpenAI first if API key is available (check both embeddings key and general OpenAI key)
    if OPENAI_EMBEDDINGS_AVAILABLE:
        api_key = (
            st.session_state.get("embeddings_openai_key") or 
            os.getenv("OPENAI_API_KEY") or 
            st.session_state.get("openai_api_key")
        )
        if api_key:
            # Create embeddings instance (we'll catch errors during actual use)
            return OpenAIEmbeddings(
                model="text-embedding-3-large",
                openai_api_key=api_key
            )
    
    # Fallback to SentenceTransformers
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        st.info("📦 Using SentenceTransformers embeddings (free, runs locally).")
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
    
    st.error("❌ No embeddings model available. Please install sentence-transformers or set a valid OPENAI_API_KEY.")
    return None


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into manageable chunks using RecursiveCharacterTextSplitter.
    Preserves metadata including page numbers.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    
    split_docs = []
    for doc in documents:
        # Split the text
        chunks = splitter.split_text(doc.page_content)
        
        # Create new Document objects with metadata
        for chunk_text in chunks:
            split_docs.append(Document(
                page_content=chunk_text,
                metadata=doc.metadata.copy()  # Preserve source, page, etc.
            ))
    
    return split_docs


def get_or_create_vectorstore(documents: List[Document], embeddings):
    """
    Create or load FAISS vector store.
    Caches embeddings to avoid re-processing PDFs.
    """
    if not documents:
        return None
    
    # Create a unique key from document contents
    doc_key = hashlib.md5(
        "".join([doc.page_content for doc in documents]).encode()
    ).hexdigest()
    
    cache_file = CACHE_DIR / f"vectorstore_{doc_key}.faiss"
    cache_pkl = CACHE_DIR / f"vectorstore_{doc_key}.pkl"
    
    # Try to load cached vector store
    if cache_file.exists() and cache_pkl.exists():
        try:
            vectorstore = FAISS.load_local(
                str(CACHE_DIR),
                embeddings,
                allow_dangerous_deserialization=True,
                index_name=f"vectorstore_{doc_key}"
            )
            st.success("Loaded cached embeddings from disk.")
            return vectorstore
        except Exception as e:
            st.warning(f"Could not load cached vectorstore: {e}. Creating new one...")
    
    # Create new vector store
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Save to cache
    try:
        vectorstore.save_local(
            str(CACHE_DIR),
            index_name=f"vectorstore_{doc_key}"
        )
        st.success("Saved embeddings to cache for future use.")
    except Exception as e:
        st.warning(f"Could not save vectorstore to cache: {e}")
    
    return vectorstore


def get_answer(vectorstore, question: str, llm):
    """
    Retrieve relevant chunks and generate answer using RAG.
    Uses LCEL (LangChain Expression Language) approach for LangChain 1.0+.
    Returns answer and source documents.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    
    # Create prompt template
    template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}

Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create chain using LCEL
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    
    # Get answer
    response = chain.invoke(question)
    answer = response.content if hasattr(response, 'content') else str(response)
    
    # Get source documents
    source_docs = retriever.get_relevant_documents(question)
    
    return answer, source_docs


def format_source_docs(source_docs: List[Document]) -> str:
    """
    Format source documents for display with filename and page numbers.
    """
    sources = []
    seen = set()
    
    for doc in source_docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        key = f"{source}_page_{page}"
        
        if key not in seen:
            seen.add(key)
            preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            sources.append(f"**{source}** (Page {page})\n{preview}")
    
    return "\n\n---\n\n".join(sources)


def main():
    """Main Streamlit application."""
    st.title("📚 Retrieval-Augmented Generation (RAG) PDF Q&A System")
    st.markdown("Upload PDF files and ask questions based on their contents.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # LLM Provider selection
        llm_provider = st.selectbox(
            "LLM Provider",
            ["OpenAI", "Google Gemini"] if (OPENAI_LLM_AVAILABLE and GEMINI_AVAILABLE) else (["OpenAI"] if OPENAI_LLM_AVAILABLE else ["Google Gemini"] if GEMINI_AVAILABLE else []),
            index=0,
            help="Choose which LLM provider to use for generating answers"
        )
        
        # API Key inputs based on provider
        if llm_provider == "OpenAI":
            openai_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Leave empty to use SentenceTransformers embeddings only",
                value=st.session_state.get("openai_api_key", "")
            )
            if openai_key:
                st.session_state["openai_api_key"] = openai_key
                os.environ["OPENAI_API_KEY"] = openai_key
            
            # LLM Model selection for OpenAI
            llm_model = st.selectbox(
                "OpenAI Model",
                ["gpt-4-turbo-preview", "gpt-4", "gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"],
                index=4,
                help="OpenAI model for generating answers"
            )
            st.session_state["llm_model"] = llm_model
            st.session_state["llm_provider"] = "openai"
        
        elif llm_provider == "Google Gemini":
            gemini_key = st.text_input(
                "Google Gemini API Key",
                type="password",
                help="Get your API key from: https://aistudio.google.com/apikey",
                value=st.session_state.get("gemini_api_key", "")
            )
            if gemini_key:
                st.session_state["gemini_api_key"] = gemini_key
                os.environ["GOOGLE_API_KEY"] = gemini_key
            
            # LLM Model selection for Gemini
            gemini_model = st.selectbox(
                "Gemini Model",
                ["gemini-pro", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash-exp"],
                index=2,
                help="Gemini model for generating answers"
            )
            st.session_state["llm_model"] = gemini_model
            st.session_state["llm_provider"] = "gemini"
            
            if not gemini_key:
                st.info("💡 **Free Gemini API**: Get your free API key from [Google AI Studio](https://aistudio.google.com/apikey)")
        
        # OpenAI API Key for embeddings (separate from LLM)
        st.divider()
        st.subheader("Embeddings (Optional)")
        embeddings_openai_key = st.text_input(
            "OpenAI API Key (for embeddings)",
            type="password",
            help="Leave empty to use SentenceTransformers (free, local)",
            value=st.session_state.get("embeddings_openai_key", "")
        )
        if embeddings_openai_key:
            st.session_state["embeddings_openai_key"] = embeddings_openai_key
            # Use embeddings key if LLM is not OpenAI
            if llm_provider != "OpenAI":
                os.environ["OPENAI_API_KEY"] = embeddings_openai_key
        
        st.divider()
        
        # Clear cache button
        if st.button("🗑️ Clear Cache", help="Clear cached embeddings"):
            import shutil
            if CACHE_DIR.exists():
                shutil.rmtree(CACHE_DIR)
                CACHE_DIR.mkdir(exist_ok=True)
            st.success("Cache cleared!")
    
    # Initialize session state
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    
    # File upload section
    st.header("📄 Upload PDF Files")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to analyze"
    )
    
    # Process uploaded files
    if uploaded_files:
        if st.button("🔄 Process Documents", type="primary"):
            with st.spinner("Processing PDFs and creating embeddings..."):
                all_documents = []
                
                # Process each file
                progress_bar = st.progress(0)
                for idx, uploaded_file in enumerate(uploaded_files):
                    # Check if file was already processed
                    file_hash = get_file_hash(uploaded_file.read())
                    uploaded_file.seek(0)  # Reset file pointer
                    
                    # Extract text from PDF
                    documents = extract_text_from_pdf(uploaded_file.read(), uploaded_file.name)
                    
                    if documents:
                        all_documents.extend(documents)
                        st.success(f"✓ Processed {uploaded_file.name} ({len(documents)} pages)")
                    else:
                        st.warning(f"⚠ Could not extract text from {uploaded_file.name}")
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                if all_documents:
                    # Split into chunks
                    with st.spinner("Splitting documents into chunks..."):
                        split_docs = split_documents(all_documents)
                        st.info(f"Created {len(split_docs)} chunks from {len(all_documents)} pages")
                    
                    # Get embeddings
                    with st.spinner("Creating embeddings..."):
                        embeddings = get_embeddings()
                        if embeddings:
                            try:
                                # Create or load vector store
                                vectorstore = get_or_create_vectorstore(split_docs, embeddings)
                                st.session_state.vectorstore = vectorstore
                                st.session_state.uploaded_files = [f.name for f in uploaded_files]
                                st.success("✅ Documents processed and ready for Q&A!")
                            except Exception as e:
                                error_msg = str(e)
                                if "401" in error_msg or "invalid_api_key" in error_msg or "AuthenticationError" in error_msg:
                                    st.error("❌ Invalid OpenAI API key. Please check your API key in the sidebar.")
                                    st.info("💡 The app will use SentenceTransformers if you remove the invalid API key.")
                                else:
                                    st.error(f"❌ Error creating embeddings: {error_msg[:200]}")
                        else:
                            st.error("❌ Failed to create embeddings. Please check your configuration.")
                else:
                    st.error("No text could be extracted from the uploaded files.")
        
        # Display uploaded files
        if st.session_state.uploaded_files:
            st.subheader("📋 Uploaded Files")
            for filename in st.session_state.uploaded_files:
                st.write(f"- {filename}")
            
            if st.button("🗑️ Clear Documents"):
                st.session_state.vectorstore = None
                st.session_state.uploaded_files = []
                st.success("Documents cleared!")
                st.rerun()
    
    st.divider()
    
    # Q&A Section
    st.header("❓ Ask Questions")
    
    if st.session_state.vectorstore is None:
        st.info("👆 Please upload and process PDF files first.")
    else:
        # Question input
        question = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="e.g., What is the main topic of the document? What are the key findings?",
            help="Ask any question about the uploaded PDF documents"
        )
        
        if st.button("🔍 Get Answer", type="primary", disabled=not question.strip()):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("🔍 Searching documents and generating answer..."):
                    # Get LLM based on provider
                    llm_provider = st.session_state.get("llm_provider", "openai")
                    llm_model = st.session_state.get("llm_model", "gpt-4o-mini")
                    
                    # Initialize LLM variable
                    llm = None
                    llm_created = False
                    
                    if llm_provider == "openai":
                        # Use OpenAI
                        if not OPENAI_LLM_AVAILABLE:
                            st.error("❌ OpenAI LLM not available. Please install langchain-community.")
                        else:
                            api_key = os.getenv("OPENAI_API_KEY") or st.session_state.get("openai_api_key")
                            if not api_key:
                                st.error("❌ OpenAI API key required. Please set it in the sidebar.")
                            else:
                                try:
                                    llm = ChatOpenAI(
                                        model=llm_model,
                                        temperature=0.0,
                                        openai_api_key=api_key
                                    )
                                    llm_created = True
                                except Exception as e:
                                    error_msg = str(e)
                                    st.error(f"❌ Error creating OpenAI LLM: {error_msg[:300]}")
                    
                    elif llm_provider == "gemini":
                        # Use Gemini
                        if not GEMINI_AVAILABLE:
                            st.error("❌ Gemini LLM not available. Please install langchain-google-genai.")
                        else:
                            api_key = os.getenv("GOOGLE_API_KEY") or st.session_state.get("gemini_api_key")
                            if not api_key:
                                st.error("❌ Gemini API key required. Please set it in the sidebar.")
                                st.info("💡 Get your free API key from: https://aistudio.google.com/apikey")
                            else:
                                try:
                                    llm = ChatGoogleGenerativeAI(
                                        model=llm_model,
                                        temperature=0.0,
                                        google_api_key=api_key
                                    )
                                    llm_created = True
                                except Exception as e:
                                    error_msg = str(e)
                                    st.error(f"❌ Error creating Gemini LLM: {error_msg[:300]}")
                    
                    # If LLM is created, proceed with answer generation
                    if llm_created and llm:
                        try:
                            answer, source_docs = get_answer(
                                st.session_state.vectorstore,
                                question,
                                llm
                            )
                                
                                # Display answer
                                st.subheader("💡 Answer")
                                st.write(answer)
                                
                                # Display sources
                                if source_docs:
                                    st.subheader("📑 Source Documents")
                                    st.markdown(format_source_docs(source_docs))
                                    
                                    # Show source details in expander
                                    with st.expander("View detailed source information"):
                                        for i, doc in enumerate(source_docs, 1):
                                            st.markdown(f"**Source {i}:**")
                                            st.markdown(f"- **File:** {doc.metadata.get('source', 'Unknown')}")
                                            st.markdown(f"- **Page:** {doc.metadata.get('page', '?')}")
                                            st.markdown(f"- **Content Preview:**")
                                            st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                                            st.divider()
                            except Exception as e:
                                error_msg = str(e)
                                if "401" in error_msg or "invalid_api_key" in error_msg or "AuthenticationError" in error_msg:
                                    st.error("❌ Invalid OpenAI API key. Please check your API key in the sidebar.")
                                    st.info("💡 Get your API key from: https://platform.openai.com/account/api-keys")
                                elif "429" in error_msg or "insufficient_quota" in error_msg or "quota" in error_msg.lower():
                                    st.error("❌ **OpenAI API Quota Exceeded**")
                                    st.warning("Your OpenAI API key has exceeded its quota or billing limit.")
                                    st.info("""
                                    **To fix this:**
                                    1. **Add credits** to your OpenAI account: https://platform.openai.com/account/billing
                                    2. **Check your usage** and billing: https://platform.openai.com/usage
                                    3. **Or remove the API key** from the sidebar to use the app with SentenceTransformers (embeddings only)
                                    
                                    **Note:** The app currently requires OpenAI API for generating answers, but embeddings can use SentenceTransformers (free).
                                    """)
                                else:
                                    st.error(f"❌ Error generating answer: {error_msg[:300]}")
                        except Exception as e:
                            error_msg = str(e)
                            if "401" in error_msg or "invalid_api_key" in error_msg or "AuthenticationError" in error_msg:
                                st.error("❌ Invalid OpenAI API key. Please check your API key in the sidebar.")
                                st.info("💡 Get your API key from: https://platform.openai.com/account/api-keys")
                            elif "429" in error_msg or "insufficient_quota" in error_msg or "quota" in error_msg.lower():
                                st.error("❌ **OpenAI API Quota Exceeded**")
                                st.warning("Your OpenAI API key has exceeded its quota or billing limit.")
                                st.info("""
                                **To fix this:**
                                1. **Add credits** to your OpenAI account: https://platform.openai.com/account/billing
                                2. **Check your usage** and billing: https://platform.openai.com/usage
                                3. **Or remove the API key** from the sidebar to use the app with SentenceTransformers (embeddings only)
                                
                                **Note:** The app currently requires OpenAI API for generating answers, but embeddings can use SentenceTransformers (free).
                                """)
                            else:
                                st.error(f"❌ Error creating LLM: {error_msg[:300]}")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        Built with LangChain, FAISS, and Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        st.error("❌ Application Error")
        st.code(str(e))
        st.code(traceback.format_exc())
        st.info("💡 Check the Streamlit Cloud logs for more details.")

