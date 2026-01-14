import os
import re
import glob
from typing import List, Dict, Any
import json
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
CONFIG = {
    "DATA_FOLDER": "data_set/law_dataset_books",
    "DB_DIRECTORY": "./chroma_db_legal_pakistani",
    "EMBEDDING_MODEL": "sentence-transformers/all-mpnet-base-v2",
    "LLM_MODEL": "mistralai/Mistral-7B-Instruct-v0.2",
    
    "INITIAL_CHUNK_SIZE": 1500,
    "INITIAL_CHUNK_OVERLAP": 200,
    "SEMANTIC_BREAKPOINT_THRESHOLD": 75, 
    "SEMANTIC_BREAKPOINT_TYPE": "percentile",
}


def _clean_text(text: str) -> str:
    text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[Page \d+\]', '', text, flags=re.IGNORECASE)
    
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    return text.strip()


def load_and_process_pdfs(folder_path: str) -> List[Any]:
    if not os.path.exists(folder_path):
        print(f"Error: Data folder '{folder_path}' not found.")
        return []

    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in '{folder_path}'.")
        return []

    print(f"Found {len(pdf_files)} PDF files. Starting to load and process...")
    all_docs = []
    for pdf_file in pdf_files:
        try:
            print(f"  - Processing: {os.path.basename(pdf_file)}")
            loader = PyMuPDFLoader(pdf_file)
            docs = loader.load()
            
            for doc in docs:
                doc.page_content = _clean_text(doc.page_content)
                doc.metadata["source_filename"] = os.path.basename(pdf_file)
            
            all_docs.extend(docs)
        except Exception as e:
            print(f"    [!] Could not process {os.path.basename(pdf_file)}: {e}")
            
    print(f"\nSuccessfully loaded and processed {len(all_docs)} pages from {len(pdf_files)} documents.")
    return all_docs


def create_hybrid_chunks_with_progress(documents: List[Any]) -> List[Any]:
    if not documents:
        return []
    print("Performing initial recursive split...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG["INITIAL_CHUNK_SIZE"],
        chunk_overlap=CONFIG["INITIAL_CHUNK_OVERLAP"],
        separators=["\n\n", "\n", ". ", " "]
    )
    initial_chunks = text_splitter.split_documents(documents)
    print(f"      Created {len(initial_chunks)} initial chunks.")

    print("Performing semantic split (this will take time)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=CONFIG["EMBEDDING_MODEL"],
        model_kwargs={'device': 'cpu'}
    )
    semantic_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_amount=CONFIG["SEMANTIC_BREAKPOINT_THRESHOLD"],
        breakpoint_threshold_type=CONFIG["SEMANTIC_BREAKPOINT_TYPE"]
    )

    batch_size = 50  
    final_chunks = []
    
    for i in tqdm(range(0, len(initial_chunks), batch_size), desc="Semantic Splitting Progress"):
        batch = initial_chunks[i:i+batch_size]
        split_batch = semantic_splitter.split_documents(batch)
        final_chunks.extend(split_batch)
        
    print(f"\n      Created {len(final_chunks)} final semantic chunks.")
    
    return final_chunks


def create_or_load_vectorstore(chunks: List[Any]) -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name=CONFIG["EMBEDDING_MODEL"],
        model_kwargs={'device': 'cpu'}
    )

    if os.path.exists(CONFIG["DB_DIRECTORY"]):
        print(f"Loading existing vector database from '{CONFIG['DB_DIRECTORY']}'...")
        vector_store = Chroma(
            persist_directory=CONFIG["DB_DIRECTORY"],
            embedding_function=embeddings
        )
    else:
        print(f"Creating new vector database at '{CONFIG['DB_DIRECTORY']}'...")
        if not chunks:
            print("No chunks provided. Cannot create vector store.")
            return None
            
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CONFIG["DB_DIRECTORY"]
        )
        print("Vector database created and persisted.")
        
    return vector_store


def chat_with_document(vector_db):
    print('Enter exit to close this')
    
    while True:
        query = input("User: ")
        if query.lower() == "exit":
            break
        
        results = vector_db.similarity_search(query, k=3)

        print("\n---  Top Relevant Sections Found ---")
        for i, doc in enumerate(results):
            page_num = doc.metadata.get("page", "Unknown")
            print(f"\n[Result {i+1} | From Page {page_num}]:")
            print('Meta Data: ', doc.metadata)
            print(doc.page_content.strip())
            print("-" * 50)
        print("\n")


def main():
    print("Checking Vector DataBase")
    
    if not os.path.exists(CONFIG["DB_DIRECTORY"]):
        documents = load_and_process_pdfs(CONFIG["DATA_FOLDER"])
        if not documents:
            print("Exiting due to lack of documents.")
            return
        chunks = create_hybrid_chunks_with_progress(documents)
        vector_store = create_or_load_vectorstore(chunks)
    else:
        vector_store = create_or_load_vectorstore(chunks=None)

    if not vector_store:
        print("Failed to create or load vector store. Exiting.")
        return

    chat_with_document(vector_store)


if __name__ == "__main__":
    main()