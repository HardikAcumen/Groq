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


# from llama_index import set_global_service_context


logging.basicConfig(filename="built.log" , level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv() 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  


"""
This is function to build pipeline

Returns : query_engine and pipeline both 

It takes path_data (Path to folder where all text and pdf data is available)
"""

logger = logging.getLogger('simpleExample')

def load_model():
    # llm = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY)
    llm = Groq(model="llama-3.1-70b-versatile", api_key=GROQ_API_KEY)
    logger.info('LLM Loaded')
    
    embed_model = HuggingFaceEmbedding("BAAI/bge-large-en-v1.5")
    logger.info('Embedding Model Loaded')

    return llm , embed_model

def create_milvus_vector_store(uri : str , dimention : int ) -> StorageContext:
    vector_store = MilvusVectorStore(uri=uri, dim=dimention, overwrite=True)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # logger.critical("created Vector Store , Storage Context from 'create_milvus_vector_store' ")
    return vector_store , storage_context

def build_pipeline(llm : Groq , embed_model : HuggingFaceEmbedding , 
                   vector_store : MilvusVectorStore ,
                   storage_context : StorageContext):
    transformations = [
        SentenceSplitter(chunk_size=500, chunk_overlap=20),
        TitleExtractor(
            llm=llm, metadata_mode=MetadataMode.EMBED, num_workers=4
        ),
        SummaryExtractor(
            llm=llm, metadata_mode=MetadataMode.EMBED, num_workers=4
        ),
        embed_model
    ]

    # logger.critical('Built Transformations' , transformations)
    
    
    documents = SimpleDirectoryReader("Data").load_data()

    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)

    docstore = SimpleDocumentStore()

    pipeline = IngestionPipeline(transformations=transformations,
                            docstore=docstore,
                            vector_store=vector_store,
                            docstore_strategy=DocstoreStrategy.UPSERTS
                            )
    
    logger.critical("Built the pipeline")

    query_engine = index.as_query_engine(llm = llm , embed_model = embed_model)

    return query_engine, pipeline, index

    

def run_pipeline(documents : list ,  pipeline : IngestionPipeline):
    nodes = pipeline.run(documents=documents)
    return nodes


llm , embed_model = load_model()

vector_store , storage_context = create_milvus_vector_store("milvus_demo.db" , 1024)

query_engine, pipeline, index = build_pipeline(llm, embed_model, vector_store= vector_store, storage_context=storage_context)
documents = SimpleDirectoryReader("Data").load_data()

Settings.embed_model = embed_model
Settings.llm = llm

nodes = pipeline.run(documents=documents)

storage_context.docstore.add_documents(nodes)
print(f"Nodes inserted: {len(nodes)}")

pipeline.persist("pipeline")



# index.storage_context.persist(persist_dir="index")

