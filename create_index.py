"""
Create a Pinecone index with dimension 1536 (for OpenAI embeddings).
The Console UI may only show 1024/2048; the API accepts 1536 — run this script.
If INDEX_NAME already exists with wrong dimension, delete it in Console first.
"""
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion, Metric

load_dotenv()

if __name__ == "__main__":
    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("INDEX_NAME")
    if not api_key or not index_name:
        raise SystemExit("Set PINECONE_API_KEY and INDEX_NAME in .env")

    pc = Pinecone(api_key=api_key)
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric=Metric.COSINE,
        spec=ServerlessSpec(
            cloud=CloudProvider.AWS,
            region=AwsRegion.US_EAST_1,
        ),
    )
    print(f"Created index {index_name} with dimension 1536.")
