from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.ingestion import (
    DocstoreStrategy,
    IngestionPipeline,
    IngestionCache,
)
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.core.base.base_query_engine import BaseQueryEngine
from redisvl.schema import IndexSchema
from llama_index.core import Settings
import os
from llama_index.llms.groq import Groq
from dotenv import load_dotenv







def read_docs(input_dir : str) -> list:
    reader = SimpleDirectoryReader(input_dir="Data")
    docs = reader.load_data()
    return docs



def generate_pipline(redis_url : str , embed_model_name : str) -> IngestionPipeline:

    embed_model = HuggingFaceEmbedding(model_name=embed_model_name)

    custom_schema = IndexSchema.from_dict(
    {
        "index": {"name": "gdrive", "prefix": "doc"},
        # customize fields that are indexed
        "fields": [
            # required fields for llamaindex
            {"type": "tag", "name": "id"},
            {"type": "tag", "name": "doc_id"},
            {"type": "text", "name": "text"},
            # custom vector field for bge-small-en-v1.5 embeddings
            {
                "type": "vector",
                "name": "vector",
                "attrs": {
                    "dims": 384,
                    "algorithm": "hnsw",
                    "distance_metric": "cosine",
                },
            },
        ],
    })


    vector_store = RedisVectorStore(
        schema=custom_schema, redis_url = redis_url
        )

    cache = IngestionCache(
    cache=RedisCache.from_host_and_port("localhost", 6379),
    collection="redis_cache",
    )

    
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(),
            embed_model,
        ],
        docstore=RedisDocumentStore.from_host_and_port(
            "localhost", 6379, namespace="document_store"
        ),
        vector_store=vector_store,
        cache=cache,
        docstore_strategy=DocstoreStrategy.UPSERTS,
    )
    return pipeline



def RAG_model(GROQ_API_KEY : str , pipeline : IngestionPipeline ,
          llm_model_name : str, embed_model : HuggingFaceEmbedding , 
          docs : list) -> BaseQueryEngine:



    llm = Groq(model=llm_model_name, api_key=GROQ_API_KEY)
    


    index = VectorStoreIndex.from_vector_store(
        pipeline.vector_store, embed_model=embed_model
    )

    query_engine = index.as_query_engine()  

    Settings.llm = llm
    Settings.embed_model = embed_model
    nodes = pipeline.run(documents=docs)

    return query_engine

"""
This function is used for loading model without any embedding docs.
The query engine retured by this will not be 
"""
def model_RAG(GROQ_API_KEY : str, pipeline : IngestionPipeline , 
            llm_model_name : str  , embed_model : HuggingFaceEmbedding )-> BaseQueryEngine:

    llm = Groq(model=llm_model_name, api_key=GROQ_API_KEY)

    index = VectorStoreIndex.from_vector_store(
        pipeline.vector_store, embed_model=embed_model
    )

    query_engine = index.as_query_engine()  
    Settings.llm = llm
    Settings.embed_model = embed_model

    return query_engine



# Main codes and Hardcode values which will be used when using this code as module.

# load_dotenv() 
# GROQ_API_KEY = os.getenv("GROQ_API_KEY") 

# llm = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY)
# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


# vector_store = RedisVectorStore(schema=custom_schema,
#     redis_url="redis://default:uGTOv5VlEb1lTMwef3rcT6KppM73it2Q@redis-10097.c8.us-east-1-4.ec2.redns.redis-cloud.com:10097"
#     )