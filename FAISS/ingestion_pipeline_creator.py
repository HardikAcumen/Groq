# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
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

from sentence_transformers import SentenceTransformer
import logging
import logging.config
from llama_index.llms.groq import Groq
from llama_index.core.schema import MetadataMode
from llama_index.core.extractors import TitleExtractor, SummaryExtractor
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
import os 
from dotenv import load_dotenv


logging.basicConfig(filename="built.log" , level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv() 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  


"""
This is function to build pipeline

Returns : query_engine and pipeline both 

It takes path_data (Path to folder where all text and pdf data is available)
"""


def build_pipeline(path_data : str):
    logger = logging.getLogger('simpleExample')

    llm = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY)
    logger.info('LLM Loaded')
    

    embed_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    # embed_model = HuggingFaceEmbedding("BAAI/bge-large-en-v1.5")
    logger.info('Embedding Model Loaded')

    transformations = [
        SentenceSplitter(chunk_size=1024, chunk_overlap=20),
        TitleExtractor(
            llm=llm, metadata_mode=MetadataMode.EMBED, num_workers=4
        ),
        SummaryExtractor(
            llm=llm, metadata_mode=MetadataMode.EMBED, num_workers=4
        ),
        embed_model
    ]

    logger.critical('Built Transformations' , transformations)

    vector_store = MilvusVectorStore(uri="./milvus_demo.db", dim=384, overwrite=True)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    documents = SimpleDirectoryReader("Data").load_data()
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    docstore = SimpleDocumentStore()
    logger.critical("Vector Store , Storage Context" , vector_store , storage_context)

    pipeline = IngestionPipeline(transformations=transformations ,
                            docstore=docstore,
                            vector_store=vector_store,
                            docstore_strategy=DocstoreStrategy.UPSERTS
                            )
    
    logger.critical("Built the pipeline" , pipeline)
    query_engine = index.as_query_engine()

    return query_engine , pipeline

    

def run_pipeline(documents : list ,  pipeline : IngestionPipeline):
    nodes = pipeline.run(documents=documents)
    return nodes


