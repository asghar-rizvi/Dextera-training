import os
import json
from typing import List
from tqdm import tqdm
from pymongo import MongoClient

from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

MONGO_URI = os.getenv("MONGO_URI")

CONFIG = {
    "MONGO_URI": MONGO_URI,
    "DB_NAME": "legal_pakistan_db",
    "COLLECTION_NAME": "vector_knowledge",
    "INDEX_NAME": "vector_index",

    "NEW_LAWS_JSON_FILE": "laws_part_2.json",  

    "EMBEDDING_MODEL": "sentence-transformers/all-mpnet-base-v2",
}


def load_new_laws() -> List[Document]:
    law_docs = []

    if not os.path.exists(CONFIG["NEW_LAWS_JSON_FILE"]):
        print(f"Error: {CONFIG['NEW_LAWS_JSON_FILE']} not found...")
        return law_docs

    with open(CONFIG["NEW_LAWS_JSON_FILE"], "r", encoding="utf-8") as f:
        laws = json.load(f)

        for law in laws:
            law_docs.append(
                Document(
                    page_content=law["content"],
                    metadata={
                        "source_type": "statute",
                        "doc_type": "law"
                    }
                )
            )

    print(f"-> Loaded {len(law_docs)} new laws from {CONFIG['NEW_LAWS_JSON_FILE']}")
    return law_docs


def update_vector_store():
    print("1. Connecting to MongoDB Atlas...")
    client = MongoClient(CONFIG["MONGO_URI"])
    collection = client[CONFIG["DB_NAME"]][CONFIG["COLLECTION_NAME"]]

    existing_count = collection.count_documents({})
    existing_statutes = collection.count_documents({"source_type": "statute"})
    existing_cases = collection.count_documents({"source_type": "case_law"})

    print(f"\n2. urrent Database Stats:")
    print(f"->Total documents: {existing_count}")
    print(f"->Statutes: {existing_statutes}")
    print(f"->Case laws: {existing_cases}")

    print(f"\n3. Loading embedding model: {CONFIG['EMBEDDING_MODEL']}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=CONFIG["EMBEDDING_MODEL"]
    )

    print(f"\n4. Loading new laws from {CONFIG['NEW_LAWS_JSON_FILE']}...")
    new_law_docs = load_new_laws()

    if not new_law_docs:
        print("No new laws to add. Exiting.")
        return

    print(f"\n5.Uploading {len(new_law_docs)} new statutes to MongoDB...")
    
    batch_size = 100
    total_uploaded = 0

    for i in tqdm(range(0, len(new_law_docs), batch_size), desc="Uploading batches"):
        batch = new_law_docs[i:i+batch_size]

        try:
            MongoDBAtlasVectorSearch.from_documents(
                documents=batch,
                embedding=embeddings,
                collection=collection,
                index_name=CONFIG["INDEX_NAME"]
            )
            total_uploaded += len(batch)
        except Exception as e:
            print(f"\nError uploading batch {i//batch_size + 1}: {str(e)}")
            continue

    new_total_count = collection.count_documents({})
    new_statutes_count = collection.count_documents({"source_type": "statute"})

    print(f"\n==========> Upload Complete ")
    print(f"\n =======>>>>  Updated Database Stats:")
    print(f"   Total documents: {new_total_count} (+{new_total_count - existing_count})")
    print(f"   Statutes: {new_statutes_count} (+{new_statutes_count - existing_statutes})")
    print(f"   Case laws: {existing_cases} (unchanged)")

    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=CONFIG["INDEX_NAME"]
    )

    return vector_store


def verify_update(vector_db):    
    test_query = input("\nEnter a test query (or press Enter to skip): ").strip()
    if not test_query:
        print("Skipping verification test.")
        return
    print(f"\nQuery: {test_query}")

    statute_results = vector_db.similarity_search_with_score(
        test_query,
        k=3,
        pre_filter={"source_type": "statute"}
    )

    print("\nTOP STATUTE RESULTS:")
    print("-" * 60)
    for i, (doc, score) in enumerate(statute_results, 1):
        print(f"\n[{i}] Score: {score:.4f}")
        print(f"Content: {doc.page_content[:250]}...")

    case_results = vector_db.similarity_search_with_score(
        test_query,
        k=2,
        pre_filter={"source_type": "case_law"}
    )

    print("\nTOP CASE LAW RESULTS:")
    print("-" * 60)
    for i, (doc, score) in enumerate(case_results, 1):
        print(f"\n[{i}] Score: {score:.4f}")
        print(f"Content: {doc.page_content[:250]}...")


if __name__ == "__main__":
    vector_store = update_vector_store()

    if vector_store:
        verify_update(vector_store)

