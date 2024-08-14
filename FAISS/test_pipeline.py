from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from sentence_transformers import SentenceTransformer
from llama_index.core.storage.docstore import SimpleDocumentStore

from llama_index.core.ingestion import (
    DocstoreStrategy,
    IngestionPipeline,
    IngestionCache,
)
from llama_index.core.node_parser import SentenceSplitter
import logging
import logging.config
from llama_index.llms.groq import Groq
from llama_index.core.schema import MetadataMode
from llama_index.core.extractors import TitleExtractor, SummaryExtractor
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import VectorStoreIndex, StorageContext , load_index_from_storage , load_indices_from_storage
import os 
from dotenv import load_dotenv
from pymilvus import MilvusClient
import pymilvus
from llama_index.core import Settings
import ingestion_pipeline_milvus as mlv 


load_dotenv() 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  

llm , embed_model = mlv.load_model()

documents = SimpleDirectoryReader("Data").load_data()


vector_store = MilvusVectorStore(uri="milvus_demo.db", overwrite=False)
print("Vector store done")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
print("Storage Done")

index = load_index_from_storage(storage_context=storage_context , embed_model=embed_model)
print("Index Done")
query_engine = index.as_query_engine( llm=llm , embed_model = embed_model)
print("Query Engine Done")


Settings.llm = llm
Settings.embed_model = embed_model

# while True:
#     query = input("Enter query: ")
#     if(query == "Q"):
#         break
#     else:
#         res = query_engine.query(query)
#         print(f"Question: {query}")
#         print(f"Answer: {res}")
#         print("")
