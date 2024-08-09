from  ingestion_pipeline_creator import run_pipeline , build_pipeline
from llama_index.core import SimpleDirectoryReader
import logging
import logging.config


logging.basicConfig(filename="test.log" , level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger('simpleExample')


# documents = SimpleDirectoryReader("Data").load_data()
# logger.info("Read the documents" , ValueError)


query_engine , pipeline = build_pipeline("Data")
logger.critical(f"Built Pipeline {pipeline}")
logger.critical(f"Built query_engine {query_engine}")

# nodes = run_pipeline(documents , pipeline)
# logger.critical(f"inserted {len(nodes)} nodes")



