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

    "CASES_JSONL_FILE": "legal_assistant_db.jsonl",
    "LAWS_JSON_FILE": "laws.json",

    "EMBEDDING_MODEL": "sentence-transformers/all-mpnet-base-v2",
}

def load_laws() -> List[Document]:
    law_docs = []

    if os.path.exists(CONFIG["LAWS_JSON_FILE"]):
        with open(CONFIG["LAWS_JSON_FILE"], "r", encoding="utf-8") as f:
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

    print(f"Loaded {len(law_docs)} laws")
    return law_docs


def load_cases() -> List[Document]:
    case_docs = []

    if os.path.exists(CONFIG["CASES_JSONL_FILE"]):
        with open(CONFIG["CASES_JSONL_FILE"], "r", encoding="utf-8") as f:
            for line in f:
                case = json.loads(line)

                content = (
                    f"Case Name: {case.get('case_name')}\n"
                    f"Summary: {case.get('summary')}"
                )

                case_docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source_type": "case_law",
                            "case_name": case.get("case_name"),
                            "filename": case.get("filename")
                        }
                    )
                )

    print(f"Loaded {len(case_docs)} cases")
    return case_docs


def get_vector_store():

    client = MongoClient(CONFIG["MONGO_URI"])
    collection = client[CONFIG["DB_NAME"]][CONFIG["COLLECTION_NAME"]]

    embeddings = HuggingFaceEmbeddings(
        model_name=CONFIG["EMBEDDING_MODEL"]
    )

    if collection.count_documents({}) == 0:
        print("Collection empty → Starting fresh ingestion...")

        law_docs = load_laws()
        case_docs = load_cases()

        all_docs = law_docs + case_docs

        print(f"Total documents to upload: {len(all_docs)}")

        batch_size = 100

        for i in tqdm(range(0, len(all_docs), batch_size), desc="Uploading batches"):
            batch = all_docs[i:i+batch_size]

            MongoDBAtlasVectorSearch.from_documents(
                documents=batch,
                embedding=embeddings,
                collection=collection,
                index_name=CONFIG["INDEX_NAME"]
            )

        print("Done...uploaded all structured data.")

    else:
        print("Already available mongodb vector db.")

    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=CONFIG["INDEX_NAME"]
    )

    return vector_store


def chat_loop(vector_db):

    print("\n--- Pakistani Legal Assistant (CLEAN DATA MODE) ---")

    while True:
        query = input("\nQuery (or 'exit'): ")
        if query.lower() == "exit":
            break

        laws = vector_db.similarity_search_with_score(
            query,
            k=3,
            pre_filter={"source_type": "statute"}
        )

        cases = vector_db.similarity_search_with_score(
            query,
            k=2,
            pre_filter={"source_type": "case_law"}
        )

        print("\n=========> STATUTES")
        for doc, score in laws:
            print(f"\nScore: {score:.4f}")
            print(doc.page_content[:300])

        print("\n=========> CASE LAW")
        for doc, score in cases:
            print(f"\nScore: {score:.4f}")
            print(doc.page_content[:300])


if __name__ == "__main__":
    vector_store = get_vector_store()
    chat_loop(vector_store)