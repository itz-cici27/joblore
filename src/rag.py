import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Set up project path constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

def build_vector_store():
    """
    Scans the docs/ folder recursively, extracts text from PDFs, applies structural metadata tags (school/federal), and builds the vector database.
    """
    print("⏳ Scanning document folders and initializing text splitters...")
    
    # 1. Initialize text chunker and open-source embedding engine
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    embedding_engine = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    all_chunks = []
    
    # 2. Walk through docs/ directories to capture files and folder structures
    if not os.path.exists(DOCS_DIR):
        raise FileNotFoundError(f"Missing documentation source folder at {DOCS_DIR}")
        
    for root, dirs, files in os.walk(DOCS_DIR):
        for file in files:
            if file.lower().endswith('.pdf'):
                file_path = os.path.join(root, file)
                
                # Determine context metadata based on the folder name
                folder_name = os.path.basename(root).lower()
                context_category = "federal" if folder_name == "federal" else folder_name
                
                print(f"📄 Processing: [{context_category.upper()}] {file}")
                
                try:
                    # Load PDF and split text into manageable paragraphs
                    loader = PyPDFLoader(file_path)
                    docs = loader.load_and_split(text_splitter)
                    
                    # Inject our architectural metadata tags into each individual text chunk
                    for doc in docs:
                        doc.metadata["source_scope"] = context_category
                        doc.metadata["file_name"] = file
                    
                    all_chunks.extend(docs)
                except Exception as e:
                    print(f"⚠️ Failed to parse {file}: {e}")

    if not all_chunks:
        print("❌ No PDF records found. Please ensure files are placed in docs/federal/ or docs/uiuc/")
        return None

    # 3. Feed the tagged chunks into our local Chroma vector database
    print(f"📦 Indexing {len(all_chunks)} text segments into local ChromaDB storage...")
    vector_db = Chroma.from_documents(
        documents=all_chunks,
        embedding=embedding_engine,
        persist_directory=DB_DIR
    )
    print("✅ Vector database successfully compiled and preserved locally!")
    return vector_db

def query_compliance_engine(school_code, user_query):
    """
    Queries the vector store using a metadata filter to guarantee a student
    only receives federal answers + their own university's specific guidance.
    """
    embedding_engine = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Reconnect to our persisted storage
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embedding_engine)
    
    # Structural Metadata Filter: Match 'federal' OR the student's selected school code
    metadata_filter = {
        "source_scope": {"$in": ["federal", school_code.lower()]}
    }
    
    print(f"\n🔍 Searching compliance context for [{school_code.upper()}] regarding: '{user_query}'...")
    
    # Fetch the top 3 closest matching context snippets matching the guidelines
    results = vector_db.similarity_search(user_query, k=3, filter=metadata_filter)
    
    return results

if __name__ == "__main__":
    # Local verification block to test ingestion and isolation rules
    try:
        # Build the DB from your folders
        db = build_vector_store()
        
        if db:
            # Test Query: Request information acting as a UIUC student
            test_query = "What are the rules for CPT application timelines?"
            matched_segments = query_compliance_engine("uiuc", test_query)
            
            print("\n💡 Top relevant context matches retrieved:")
            for i, chunk in enumerate(matched_segments, 1):
                print(f"\n--- Match #{i} (Scope: {chunk.metadata['source_scope'].upper()} | File: {chunk.metadata['file_name']}) ---")
                print(chunk.page_content[:300] + "...")
                
    except Exception as e:
        print(f"❌ Error encountered: {e}")