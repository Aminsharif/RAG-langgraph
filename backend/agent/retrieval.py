"""Manage the configuration of various retrievers.

This module provides functionality to create and manage retrievers for different
vector store backends, specifically Elasticsearch, Pinecone, and MongoDB.

The retrievers support filtering results by user_id to ensure data isolation between users.
"""

import os
from contextlib import contextmanager
from typing import Generator

from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStoreRetriever

from backend.agent.configuration import Configuration, IndexConfiguration
from backend.agent.embeddings import get_embeddings_model

from langchain_weaviate import WeaviateVectorStore
from backend.agent.constants import WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX
import weaviate

## Encoder constructors


def make_text_encoder(model: str) -> Embeddings:
    """Connect to the configured text encoder."""
    provider, model = model.split("/", maxsplit=1)
    match provider:
        case "openai":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model=model)
        case "cohere":
            from langchain_cohere import CohereEmbeddings

            return CohereEmbeddings(model=model)  # type: ignore
        case _:
            raise ValueError(f"Unsupported embedding provider: {provider}")


## Retriever constructors


@contextmanager
def make_elastic_retriever(
    configuration: IndexConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific elastic index."""
    from langchain_elasticsearch import ElasticsearchStore

    connection_options = {}
    if configuration.retriever_provider == "elastic-local":
        connection_options = {
            "es_user": os.environ["ELASTICSEARCH_USER"],
            "es_password": os.environ["ELASTICSEARCH_PASSWORD"],
        }

    else:
        connection_options = {"es_api_key": os.environ["ELASTICSEARCH_API_KEY"]}

    vstore = ElasticsearchStore(
        **connection_options,  # type: ignore
        es_url=os.environ["ELASTICSEARCH_URL"],
        index_name="langchain_index",
        embedding=embedding_model,
    )

    search_kwargs = configuration.search_kwargs

    search_filter = search_kwargs.setdefault("filter", [])
    search_filter.append({"term": {"metadata.user_id": configuration.user_id}})
    yield vstore.as_retriever(search_kwargs=search_kwargs)


@contextmanager
def make_pinecone_retriever(
    configuration: IndexConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific pinecone index."""
    from langchain_pinecone import PineconeVectorStore

    search_kwargs = configuration.search_kwargs

    search_filter = search_kwargs.setdefault("filter", {})
    search_filter.update({"user_id": configuration.user_id})
    vstore = PineconeVectorStore.from_existing_index(
        os.environ["PINECONE_INDEX_NAME"], embedding=embedding_model
    )
    yield vstore.as_retriever(search_kwargs=search_kwargs)


@contextmanager
def make_mongodb_retriever(
    configuration: IndexConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific MongoDB Atlas index & namespaces."""
    from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch

    vstore = MongoDBAtlasVectorSearch.from_connection_string(
        os.environ["MONGODB_URI"],
        namespace="langgraph_retrieval_agent.default",
        embedding=embedding_model,
    )
    search_kwargs = configuration.search_kwargs
    pre_filter = search_kwargs.setdefault("pre_filter", {})
    pre_filter["user_id"] = {"$eq": configuration.user_id}
    yield vstore.as_retriever(search_kwargs=search_kwargs)

# @contextmanager
# def make_weaviate_retriever(
#     configuration: IndexConfiguration, embedding_model: Embeddings
# ) -> Generator[VectorStoreRetriever, None, None]:
#     """Configure this agent to connect to a specific Weaviate index."""
#     try:
#         import weaviate
#         from weaviate import Client
#         from weaviate.classes.init import Auth
#         from weaviate.classes.query import Filter

#         from langchain_weaviate.vectorstores import WeaviateVectorStore
#     except ImportError:
#         raise ImportError(
#             "Weaviate dependencies not installed. "
#             "Please install langchain-weaviate and weaviate-client"
#         )
    
#     # Get Weaviate connection parameters from environment
#     weaviate_url = os.environ.get("WEAVIATE_URL")
#     weaviate_api_key = os.environ.get("WEAVIATE_API_KEY")
    
#     if not weaviate_url:
#         raise ValueError("WEAVIATE_URL environment variable not set")
#     if not weaviate_api_key:
#         raise ValueError("WEAVIATE_API_KEY environment variable not set")
    
#     # Define index name
#     WEAVIATE_INDEX_NAME = WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX_NAME
    
#     # Connect to Weaviate
#     client = weaviate.connect_to_weaviate_cloud(
#         cluster_url=weaviate_url,
#         auth_credentials=Auth.api_key(weaviate_api_key),
#     )
    
#     try:
#         # Create vector store
#         store = WeaviateVectorStore(
#             client=client,
#             index_name=WEAVIATE_INDEX_NAME,
#             text_key="text",  # or "content" based on your schema
#             embedding=embedding_model,  # Use the provided embedding model
#             attributes=["source", "title", "user_id"],  # Still include attributes for metadata
#         )
        
#         # Use the search kwargs as provided in configuration
#         # No user_id filtering applied
#         search_kwargs = configuration.search_kwargs.copy()
        
#         # Yield the retriever
#         yield store.as_retriever(search_kwargs=search_kwargs)
        
#     finally:
#         # Clean up the Weaviate connection
#         client.close()
@contextmanager
def make_weaviate_retriever(
    configuration: IndexConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific Weaviate index."""
    try:
        import weaviate
        from weaviate.classes.init import Auth
        from weaviate.classes.query import Filter
        from langchain_weaviate.vectorstores import WeaviateVectorStore
    except ImportError:
        raise ImportError(
            "Weaviate dependencies not installed. "
            "Please install langchain-weaviate and weaviate-client"
        )

    weaviate_url = os.environ.get("WEAVIATE_URL")
    weaviate_api_key = os.environ.get("WEAVIATE_API_KEY")

    if not weaviate_url:
        raise ValueError("WEAVIATE_URL environment variable not set")
    if not weaviate_api_key:
        raise ValueError("WEAVIATE_API_KEY environment variable not set")

    WEAVIATE_INDEX_NAME = WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
    )

    try:
        store = WeaviateVectorStore(
            client=client,
            index_name=WEAVIATE_INDEX_NAME,
            text_key="text",
            embedding=embedding_model,
            attributes=["source", "title", "user_id"],
        )

        search_kwargs = configuration.search_kwargs.copy()

        # ✅ ADD USER FILTER HERE
        if configuration.user_id:
            search_kwargs["filters"] = Filter.by_property("user_id").equal(
                configuration.user_id
            )

        yield store.as_retriever(search_kwargs=search_kwargs)

    finally:
        client.close()

@contextmanager
def make_retriever(
    config: RunnableConfig,
) -> Generator[VectorStoreRetriever, None, None]:
    """Create a retriever for the agent, based on the current configuration."""
    configuration = IndexConfiguration.from_runnable_config(config)
    embedding_model = get_embeddings_model() #make_text_encoder(configuration.embedding_model)
    user_id = configuration.user_id
    if not user_id:
        raise ValueError("Please provide a valid user_id in the configuration.")
    match configuration.retriever_provider:
        case "elastic" | "elastic-local":
            with make_elastic_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case "pinecone":
            with make_pinecone_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case "mongodb":
            with make_mongodb_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case "weaviate":
            with make_weaviate_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case _:
            raise ValueError(
                "Unrecognized retriever_provider in configuration. "
                f"Expected one of: {', '.join(Configuration.__annotations__['retriever_provider'].__args__)}\n"
                f"Got: {configuration.retriever_provider}"
            )