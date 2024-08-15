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
# from llama_index.vector_stores import SimpleVectorStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core import VectorStoreIndex, StorageContext , load_index_from_storage , load_indices_from_storage
import os 
from dotenv import load_dotenv
from pymilvus import MilvusClient
import pymilvus
from llama_index.core import Settings
import ingestion_pipeline_milvus as mlv 
from llama_index.core import SummaryIndex
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.postprocessor.mixedbreadai_rerank import MixedbreadAIRerank

load_dotenv() 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  
llm = Groq(model="llama-3.1-70b-versatile", api_key=GROQ_API_KEY)
embed_model = HuggingFaceEmbedding("BAAI/bge-large-en-v1.5")
MIXEDBREADAI_API_KEY = os.getenv("MIXEDBREADAI_API_KEY")

documents = SimpleDirectoryReader("Data").load_data()

nodes = SentenceSplitter().get_nodes_from_documents(documents)
print("Nodes Done")
docstore=SimpleDocumentStore.from_persist_dir(persist_dir="index")
print("DocStore Done")

storage_context = StorageContext.from_defaults(docstore=docstore)
index = SummaryIndex(nodes, storage_context=storage_context)
print("Index Done")

mixedbreadai_rerank = MixedbreadAIRerank(
    api_key=MIXEDBREADAI_API_KEY,
    top_n=2,
    model="mixedbread-ai/mxbai-rerank-large-v1",
)
print("Reranker done")

query_engine = index.as_query_engine(
    similarity_top_k=5,
    llm=llm , 
    embed_model = embed_model
    # node_postprocessors=[mixedbreadai_rerank],  -> this rerank
)
print("Query Engine Done")


Settings.llm = llm
Settings.embed_model = embed_model

while True:
    query = input("Enter query: ")
    if(query == "Q" or query == "q"):
        break
    else:
        res = query_engine.query(query)
        print(f"Question: {query}")
        print(f"Answer: {res}")
        print(f"meta data: {res.source_nodes}")
        print(f"extra infor : {res.metadata}")
        print("")

# load_dotenv() 
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")  
# llm = Groq(model="llama-3.1-70b-versatile", api_key=GROQ_API_KEY)
# embed_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
# MIXEDBREADAI_API_KEY = os.getenv("MIXEDBREADAI_API_KEY")


# documents = SimpleDirectoryReader("Data").load_data()


# client = MilvusClient("milvus_demo.db")

# print("Vector Store Done")

# Settings.llm = llm
# Settings.embed_model = embed_model

# while True:
#     query = input("Enter query: ")
#     if(query == "Q" or query == "q"):
#         break
#     else:
#         query_embedding = embed_model.encode(query)
        
        
