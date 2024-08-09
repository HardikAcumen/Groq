import pandas as pd
import os
from llama_index.embeddings.gemini import GeminiEmbedding
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import requests
from io import StringIO
import pandas as pd
import faiss

load_dotenv() 
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")  
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY



def get_data_from_internet(url) -> pd.DataFrame:
    response = requests.get(url)
    data = pd.read_csv(StringIO(response.text), sep='\t')

    return data

def get_embeddings_values(data: pd.DataFrame , model_name : str) -> dict:
    
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


def mappings_to_dataframe(embeddings_dict: dict) -> pd.DataFrame:
    sentences = list(embeddings_dict.keys())
    embeddings = list(embeddings_dict.values())
    
    flattened_embeddings = [embedding[0] if isinstance(embedding, list) else embedding for embedding in embeddings]
    
    df = pd.DataFrame(flattened_embeddings, index=sentences)
    
    embedding_size = len(df.iloc[0])
    df.columns = [f'embedding_{i}' for i in range(embedding_size)]
    
    return df

def put_embeddings(embeddings : pd.DataFrame , model : str) -> tuple:
    embedding_dimentions = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dimentions)
    index.add(embeddings)
    model = SentenceTransformer(str)
    return (index , model)

def get_query( index , model , top_n : int , text : str ,):
    text_enc = model.encode([text])
    distance, index_in_data = index.search(text_enc, top_n)  
    return index_in_data

url = 'https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/sick2014/SICK_train.txt'
data = get_data_from_internet(url)
data = data[ : 5]
data = data[['sentence_A']]
model_name = "models/embedding-001"
mappings = get_embeddings_values(data , model_name)
embeddings = mappings_to_dataframe(mappings)

model = "all-MiniLM-L6-v2"
indexed_model = put_embeddings(embeddings , model)
index , model = indexed_model


result_of_search = get_query( index , model , top_n = 2 , text = "The") 

result_string = data['sentence_A'].iloc[[2 ,3]]
print(result_string)


