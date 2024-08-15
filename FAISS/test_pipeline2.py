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
embed_model = SentenceTransformer("BAAI/bge-large-en-v1.5")


documents = SimpleDirectoryReader("Data").load_data()

vector_store = MilvusVectorStore(uri=uri, dim=dimention, overwrite=True , 
                                     collection_name = "llama" ,
                                     hybrid_ranker="RRFRanker", 
                                     hybrid_ranker_params={"k": 60} 
                                    )

vector_store = MilvusVectorStore(
    dim=1024,
    overwrite=True,
    enable_sparse=True,
    hybrid_ranker="RRFRanker",
    hybrid_ranker_params={"k": 60},
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)
print("Vector Store Done")


Settings.llm = llm
Settings.embed_model = embed_model

while True:
    query = input("Enter query: ")
    if(query == "Q" or query == "q"):
        break
    else:
        query_embedding = embed_model.encode(query)
        
        
