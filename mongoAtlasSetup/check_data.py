from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()



MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "legal_pakistan_db"
COLLECTION_NAME = "vector_knowledge"
INDEX_NAME = "vector_index"

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
TOP_K = 5
THRESHOLD = 0.8 

client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]




embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    index_name=INDEX_NAME,
)





def run_query(query):

    print("\nQUERY:", query)
    print("=" * 60)
    print("\nSTATUTE RESULTS")
    print("-" * 60)

    law_results = vector_store.similarity_search_with_score(
        query,
        k=TOP_K,
        pre_filter={"source_type": "statute"}
    )

    for i, (doc, score) in enumerate(law_results):
        print(f"\n[{i+1}] SCORE: {score:.4f}")
        print(doc.page_content[:400])

    print("\nCASE LAW RESULTS")
    print("-" * 60)

    case_results = vector_store.similarity_search_with_score(
        query,
        k=TOP_K,
        pre_filter={"source_type": "case_law"}
    )

    for i, (doc, score) in enumerate(case_results):
        print(f"\n[{i+1}] SCORE: {score:.4f}")
        print(doc.page_content[:400])


if __name__ == "__main__":

    while True:
        q = input("\nEnter query (or 'exit'): ")

        if q.lower() == "exit":
            break

        run_query(q)