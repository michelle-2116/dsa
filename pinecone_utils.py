import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()
NAMESPACE="chat"
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("llm-context-index")
EMBED_MODEL = "llama-text-embed-v2"

def embed_text(text, input_type):
    response = pc.inference.embed(
        model=EMBED_MODEL,
        inputs=[text],
        parameters={"input_type": input_type}
    )
    return response.data[0].values

def upsert_message(msg_id, text, turn_id, role):
    vector = embed_text(text, input_type="passage")
    index.upsert(
        vectors=[{
            "id": str(msg_id),
            "values": vector,
            "metadata": {
                "turn_id": turn_id, 
                "role": role
            }
        }],
        namespace=NAMESPACE
    )

def query_similar_turns(text, threshold=0.40, top_k=10):
    query_vector = embed_text(text, input_type="query")
    result = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        namespace=NAMESPACE
    )

    if not result or not result.matches:
        return []

    #Extract turn_id from metadata to ensure we get the full context pair
    return [
        int(match.metadata["turn_id"])
        for match in result.matches
        if match.score >= threshold
    ]