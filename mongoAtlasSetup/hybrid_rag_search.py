from pymongo import MongoClient
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "legal_pakistan_db"
COLLECTION_NAME = "vector_knowledge"

TOP_K_BM25 = 20
TOP_K_VECTOR = 20
FINAL_TOP_K = 5

print("Loading models...")
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]





print("Loading documents from MongoDB...")
all_docs = {
    "statute": {"docs": [], "texts": []},
    "case_law": {"docs": [], "texts": []}
}

cursor = collection.find({})

for doc in cursor:
    text = doc.get("text")
    embedding = doc.get("embedding")
    source_type = doc.get("source_type", "unknown")
    
    if not text or embedding is None:
        continue
    
    if source_type not in ["statute", "case_law"]:
        continue
    
    doc_data = {
        "text": text,
        "embedding": np.array(embedding),
        "_id": str(doc["_id"]),
        "source_type": source_type,
        "doc_type": doc.get("doc_type", "unknown")
    }
    
    all_docs[source_type]["docs"].append(doc_data)
    all_docs[source_type]["texts"].append(text)

print(f"Loaded {len(all_docs['statute']['docs'])} statute documents")
print(f"Loaded {len(all_docs['case_law']['docs'])} case law documents")




def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())




print("Building BM25 indices...")
bm25_indices = {}

for source_type in ["statute", "case_law"]:
    texts = all_docs[source_type]["texts"]
    
    if len(texts) > 0:
        tokenized_corpus = [tokenize(t) for t in texts]
        bm25_indices[source_type] = BM25Okapi(tokenized_corpus)
    else:
        bm25_indices[source_type] = None

print("Setup complete!\n")




def normalize_scores(scores: np.ndarray) -> np.ndarray:
    if len(scores) == 0:
        return scores
    
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    if max_score == min_score:
        return np.ones_like(scores)
    
    normalized = (scores - min_score) / (max_score - min_score)
    return normalized

def sigmoid_normalize(scores: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-scores))

def hybrid_search(
    query: str,
    source_type: Optional[str] = None,
    top_k_bm25: int = TOP_K_BM25,
    top_k_vector: int = TOP_K_VECTOR,
    final_top_k: int = FINAL_TOP_K,
    normalize_method: str = "minmax"  
) -> List[Dict]:
    
    query_tokens = tokenize(query)
    query_embedding = embedding_model.encode(query)
    
    results_by_type = {}
    
    search_types = [source_type] if source_type else ["statute", "case_law"]
    
    for s_type in search_types:
        docs = all_docs[s_type]["docs"]
        
        if len(docs) == 0:
            continue
        
  
  
        bm25 = bm25_indices[s_type]
        bm25_scores = bm25.get_scores(query_tokens)
        bm25_top_idx = np.argsort(bm25_scores)[::-1][:top_k_bm25]
    
        embeddings = np.array([d["embedding"] for d in docs])
        vector_scores = cosine_similarity([query_embedding], embeddings)[0]
        vector_top_idx = np.argsort(vector_scores)[::-1][:top_k_vector]
        
        candidate_idx = list(set(bm25_top_idx).union(set(vector_top_idx)))
        candidates = [docs[idx] for idx in candidate_idx]
        
        if len(candidates) > 0:
            pairs = [(query, doc["text"]) for doc in candidates]
            rerank_scores = np.array(reranker.predict(pairs))
            
            if normalize_method == "sigmoid":
                rerank_scores = sigmoid_normalize(rerank_scores)
            else:  
                rerank_scores = normalize_scores(rerank_scores)
            
            ranked = sorted(
                zip(candidates, rerank_scores),
                key=lambda x: x[1],
                reverse=True
            )
            
            results_by_type[s_type] = [
                {
                    "id": doc["_id"],
                    "text": doc["text"],
                    "source_type": doc["source_type"],
                    "doc_type": doc["doc_type"],
                    "score": float(score)
                }
                for doc, score in ranked[:final_top_k]
            ]
    
    if len(search_types) > 1:
        all_results = []
        for s_type in search_types:
            if s_type in results_by_type:
                all_results.extend(results_by_type[s_type])
        
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:final_top_k]
    
    return results_by_type.get(search_types[0], [])


def display_results(results: List[Dict], title: str = "RESULTS"):
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}\n")
    
    if not results:
        print("No results found.\n")
        return
    
    for i, r in enumerate(results, 1):
        print(f"[{i}] SCORE: {r['score']:.4f}")
        print(f"    Type: {r['source_type']} | {r['doc_type']}")
        print(f"    ID: {r['id']}")
        print(f"    Text: {r['text'][:300]}...")
        print()

def run_query(query: str, normalize_method: str = "minmax"):    
    print(f"\nQUERY: {query}")
    print(f"Normalization: {normalize_method}")
    print("=" * 80)
    
    statute_results = hybrid_search(
        query, 
        source_type="statute", 
        final_top_k=FINAL_TOP_K,
        normalize_method=normalize_method
    )
    display_results(statute_results, "STATUTE RESULTS")
    
    case_results = hybrid_search(
        query, 
        source_type="case_law", 
        final_top_k=FINAL_TOP_K,
        normalize_method=normalize_method
    )
    display_results(case_results, "CASE LAW RESULTS")
    
    combined_results = hybrid_search(
        query, 
        source_type=None, 
        final_top_k=FINAL_TOP_K,
        normalize_method=normalize_method
    )
    display_results(combined_results, "COMBINED RESULTS (All Types)")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("HYBRID SEARCH READY (with normalized scores 0-1)")
    print("="*80)
    print("Options:")
    print("  - Enter a query to search")
    print("  - Type 'minmax' or 'sigmoid' to change normalization")
    print("  - Type 'exit' to quit")
    print("="*80)
    
    normalize_method = "minmax" 
    
    while True:
        query = input("\nEnter query: ").strip()
        
        if query.lower() == "exit":
            print("Goodbye!")
            break
        
        if query.lower() in ["minmax", "sigmoid"]:
            normalize_method = query.lower()
            print(f"Normalization method set to: {normalize_method}")
            continue
        
        if not query:
            print("Please enter a valid query.")
            continue
        
        run_query(query, normalize_method)