import os
import json
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

CONFIG = {
    "DB_DIRECTORY": "./chroma_db_legal_pakistani", 
    "EMBEDDING_MODEL": "sentence-transformers/all-mpnet-base-v2",
    "CASES_JSONL_FILE": "legal_assistant_db.jsonl", 
    "NUM_LAW_RESULTS": 3,
    "NUM_CASE_RESULTS": 3,
}

def load_cases_from_jsonl(file_path: str) -> List[Dict[str, Any]]:
    cases = []
    if not os.path.exists(file_path):
        print(f"Error: Cases file not found at '{file_path}'")
        return []
    
    print(f"Loading cases from '{file_path}'...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error: Could not decode line {line_number}: {e}")
    return cases

def create_case_documents(cases_data: List[Dict[str, Any]]) -> List[Document]:
    case_documents = []
    print("Transforming case data into document format...")
    for case in cases_data:
        content = f"Case Name: {case.get('case_name', 'N/A')}\n\nSummary: {case.get('summary', 'N/A')}"
        
        metadata = {
            "source": "case_law",
            "case_name": case.get("case_name")
        }
        
        case_documents.append(Document(page_content=content, metadata=metadata))
        
    print(f"Created {len(case_documents)} case documents.")
    return case_documents

def update_vector_store_with_cases(vector_store: Chroma, case_documents: List[Document]):
    if not case_documents:
        print("No case documents to add.")
        return
        
    print(f"Adding {len(case_documents)} new cases to the vector database...")
    vector_store.add_documents(case_documents)
    print("Vector database successfully updated.")

def chat_with_legal_assistant(vector_db: Chroma):
    print('\n--- Pakistani Legal Assistant ---')
    print('Enter your query, or type "exit" to close.')
    
    THRESHOLD = 0.7
    
    while True:
        query = input("\nUser Query: ")
        if query.lower() == "exit":
            break
        
        print("\nSearching for relevant legal statutes...")
        law_results_with_scores = vector_db.similarity_search_with_score(
            query, 
            k=CONFIG["NUM_LAW_RESULTS"],
            filter={"source": {"$ne": "case_law"}}
        )

        print("Searching for relevant past cases...")
        case_results_with_scores = vector_db.similarity_search_with_score(
            query, 
            k=CONFIG["NUM_CASE_RESULTS"],
            filter={"source": "case_law"}
        )

        valid_laws = [doc for doc, score in law_results_with_scores if score <= THRESHOLD]
        
        valid_cases = [doc for doc, score in case_results_with_scores if score <= THRESHOLD]

        print("\n--- Top Relevant Law Sections ---")
        if not valid_laws:
            print(f"No relevant law sections found above threshold (Score <= {THRESHOLD}).")
        else:
            for i, doc in enumerate(valid_laws):
                source_file = doc.metadata.get("source_filename", "Unknown File")
                page_num = doc.metadata.get("page", "Unknown Page")
                print(f"\n[Law Result {i+1} | From: {source_file}, Page {page_num}]")
                print(doc.page_content.strip()[:400] + "...") 
                print("-" * 40)

        print("\n--- Top Relevant Past Cases ---")
        if not valid_cases:
            print(f"No relevant past cases found above threshold (Score <= {THRESHOLD}).")
        else:
            for i, doc in enumerate(valid_cases):
                case_name = doc.metadata.get("case_name", "Unknown Case")
                print(f"\n[Case Result {i+1} | Case Name: {case_name}]")
                print(doc.page_content.strip())
                print("-" * 40)
        
        print("\n" + "="*50 + "\n")


def main():
    if not os.path.exists(CONFIG["DB_DIRECTORY"]):
        print(f"Error: Vector database not found at '{CONFIG['DB_DIRECTORY']}'.")
        print("Please run your initial script to create the database first.")
        return

    print("Loading existing vector database...")
    embeddings = HuggingFaceEmbeddings(
        model_name=CONFIG["EMBEDDING_MODEL"],
        model_kwargs={'device': 'cpu'}
    )
    vector_store = Chroma(
        persist_directory=CONFIG["DB_DIRECTORY"],
        embedding_function=embeddings
    )
    print("Vector database loaded successfully.")

    # cases_data = load_cases_from_jsonl(CONFIG["CASES_JSONL_FILE"])
    # if cases_data:
    #     case_documents = create_case_documents(cases_data)
    #     update_vector_store_with_cases(vector_store, case_documents)

    chat_with_legal_assistant(vector_store)


if __name__ == "__main__":
    main()