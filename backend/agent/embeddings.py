from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

def get_embeddings_model() -> Embeddings:
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") 