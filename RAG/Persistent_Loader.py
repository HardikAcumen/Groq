from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.ingestion import (
    DocstoreStrategy,
    IngestionPipeline,
    IngestionCache,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
import faiss

import os
from dotenv import load_dotenv

from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)

from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex


def persistent_loader(data_dir : str):

    embed_model = GeminiEmbedding(
    model_name=model_name, api_key=GOOGLE_API_KEY, title="this is a document for sample"
    )

import pandas as pd


def get_embeddings(data: pd.DataFrame) -> dict:
    model_name = "models/embedding-001"
    embed_model = GeminiEmbedding(
        model_name=model_name, api_key=GOOGLE_API_KEY, title="this is a document for sample"
    )
    
    ans = {} 
    for i, row in data.iterrows():
        sent = row['sentence_A']
        embedding = embed_model.get_text_embedding(row['sentence_A'])
        if sent not in ans:  
            ans[sent] = []
        ans[sent].append(embedding)  
    
    return ans


def embeddings_to_dataframe(embeddings_dict: dict) -> pd.DataFrame:
    sentences = list(embeddings_dict.keys())
    embeddings = list(embeddings_dict.values())
    
    flattened_embeddings = [embedding[0] if isinstance(embedding, list) else embedding for embedding in embeddings]
    
    df = pd.DataFrame(flattened_embeddings, index=sentences)
    
    embedding_size = len(df.iloc[0])
    df.columns = [f'embedding_{i}' for i in range(embedding_size)]
    
    return df




