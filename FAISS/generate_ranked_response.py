import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
from llama_index.core.postprocessor import LLMRerank
from dotenv import load_dotenv
import os
from llama_index.postprocessor.mixedbreadai_rerank import MixedbreadAIRerank
from llama_index.core import (
    StorageContext,
)

from llama_index.core import load_index_from_storage

from llama_index.vector_stores.faiss import FaissVectorStore

logging.basicConfig(filename="rerank.log" , level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv() 
MixedBreadAi_API_KEY = os.getenv("MIXEDBREADAI_API_KEY")  

"""
This is a function for creating Ranked query_engine.
It is using mixedbreadAi for reranking.
Returns : query_engine object which takes persist faiss db as input."""

def ranked_query_engine():
    
    vector_store = FaissVectorStore.from_persist_dir("./storage")
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir="./storage"
    )
    index = load_index_from_storage(storage_context=storage_context)

    mixedbreadai_rerank = MixedbreadAIRerank(
        api_key=MixedBreadAi_API_KEY,
        top_n=2,
        model="mixedbread-ai/mxbai-rerank-large-v1",
    )

    query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[mixedbreadai_rerank]
    )

    return query_engine
