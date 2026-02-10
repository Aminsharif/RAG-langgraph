from enum import Enum
from typing import Union
from langchain.embeddings.base import Embeddings
from langchain_weaviate import WeaviateVectorStore
from langchain_community.vectorstores import Chroma
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain_elasticsearch.vectorstores import ElasticsearchStore
import os
import weaviate

class VectorStoreType(Enum):
    """Supported vector store types"""
    WEAVIATE = "weaviate"
    CHROMA = "chroma"
    PINECONE = "pinecone"
    ELASTICSEARCH = "elasticsearch"
    FAISS = "faiss"


class VectorStoreFactory:
    """Factory for creating vector stores"""
    
    @staticmethod
    def create_vector_store(
        store_type: VectorStoreType,
        embedding_model: Embeddings,
        **config
    ) -> Union[WeaviateVectorStore, Chroma, PineconeVectorStore, 
               ElasticsearchStore,]:
        """Create a vector store instance"""
        WEAVIATE_URL = os.environ["WEAVIATE_URL"]
        WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
        if store_type == VectorStoreType.WEAVIATE:
            client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=weaviate.classes.init.Auth.api_key(WEAVIATE_API_KEY),
            skip_init_checks=True,
            )
            return WeaviateVectorStore(
                client=client,
                embedding=embedding_model,
                text_key="text",
                index_name=config.get("index_name", "Documents")
            )
        
        elif store_type == VectorStoreType.CHROMA:
            return Chroma(
                embedding_model=embedding_model,
                persist_directory=config.get("persist_directory", "./chroma_db"),
                collection_name=config.get("collection_name", "documents")
            )
        
        elif store_type == VectorStoreType.PINECONE:
            return PineconeVectorStore(
                embedding_model=embedding_model,
                api_key=config["api_key"],
                environment=config["environment"],
                index_name=config.get("index_name", "documents"),
                namespace=config.get("namespace", "default")
            )
        
        elif store_type == VectorStoreType.ELASTICSEARCH:
            return ElasticsearchStore(
                embedding_model=embedding_model,
                elasticsearch_url=config["elasticsearch_url"],
                username=config.get("username"),
                password=config.get("password"),
                api_key=config.get("api_key"),
                index_name=config.get("index_name", "documents")
            )
        
        # elif store_type == VectorStoreType.FAISS:
        #     return FAISSVectorStore(
        #         embedding_model=embedding_model,
        #         index_path=config.get("index_path", "./faiss_index"),
        #         dimension=config.get("dimension", 1536)
        #     )
        
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}")