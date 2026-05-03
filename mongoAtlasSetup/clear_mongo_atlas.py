from pymongo import MongoClient
import os

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "legal_pakistan_db"
COLLECTION_NAME = "vector_knowledge"

client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]

count_before = collection.count_documents({})
print(f"Documents before delete: {count_before}")
result = collection.delete_many({})

print(f"Deleted documents: {result.deleted_count}")

count_after = collection.count_documents({})
print(f"Documents after delete: {count_after}")
