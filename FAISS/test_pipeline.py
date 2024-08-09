from  FAISS.ingestion_pipeline_faiss import run_pipeline , build_pipeline , load_model
from llama_index.core import SimpleDirectoryReader
import logging
import logging.config
print("Hii")


# logging.basicConfig(filename="test.log" , level=logging.DEBUG,
#                     format='%(asctime)s - %(levelname)s - %(message)s')

# logger = logging.getLogger('simpleExample')


# # documents = SimpleDirectoryReader("Data").load_data()
# # logger.info("Read the documents" , ValueError)

# llm , embed_model = load_model()
# query_engine , pipeline = build_pipeline(llm , embed_model)


# nodes = run_pipeline(documents , pipeline)
# logger.critical(f"inserted {len(nodes)} nodes")



