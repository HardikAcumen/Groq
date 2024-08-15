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
from llama_index.core import VectorStoreIndex, StorageContext
import os 
from dotenv import load_dotenv
from pymilvus import MilvusClient
import pymilvus
from llama_index.core import Settings
import ingestion_pipeline_milvus as mlv


# If you are running on you machine please create a free token here you will also get milvus_uri
# https://cloud.zilliz.com/signup
# 
milvus_uri = "https://in03-f32579abd0b9461.api.gcp-us-west1.zillizcloud.com"
access_token = "3731c968b4b98e577286c4c076636dd7e6aaf2d740f446c031e69f5f0ebd577abeea4e40a42502d6bcd7c9a586d65ede6308e602"
# Loading models and embed model
llm , embed_model = mlv.load_model()

vector_store , storage_context = mlv.create_milvus_vector_store(uri=milvus_uri ,access_token = access_token , dimention = 1024)

query_engine, pipeline, index = mlv.build_pipeline(llm, embed_model, vector_store= vector_store, storage_context=storage_context)
documents = SimpleDirectoryReader("Data").load_data()

# Setting Embed Model
Settings.embed_model = embed_model
Settings.llm = llm

# getting nodes from run of pipeline
nodes = pipeline.run(documents=documents)
print(f"Docs inserted : {len(nodes)}")

# persisting Databases and Storage Context 
storage_context.docstore.add_documents(nodes)
storage_context.persist(persist_dir="index")
pipeline.persist("pipeline")