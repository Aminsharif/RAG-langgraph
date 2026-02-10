from elasticsearch import Elasticsearch
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from typing import List, Dict, Any, Optional
from backend.ingest.addapter.base_vector_store import BaseVectorStore
import json


class ElasticsearchVectorStore(BaseVectorStore):
    """Elasticsearch vector store implementation"""
    
    def __init__(
        self, 
        embedding_model: Embeddings,
        elasticsearch_url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        api_key: Optional[str] = None,
        index_name: str = "documents",
        ssl_verify: bool = True
    ):
        super().__init__(embedding_model)
        self.elasticsearch_url = elasticsearch_url
        self.index_name = index_name
        
        # Setup connection
        if api_key:
            self.client = Elasticsearch(
                elasticsearch_url,
                api_key=api_key,
                verify_certs=ssl_verify
            )
        elif username and password:
            self.client = Elasticsearch(
                elasticsearch_url,
                basic_auth=(username, password),
                verify_certs=ssl_verify
            )
        else:
            self.client = Elasticsearch(
                elasticsearch_url,
                verify_certs=ssl_verify
            )
        
        # Create index if not exists
        self._create_index()
    
    def _create_index(self):
        """Create Elasticsearch index with proper mappings"""
        if not self.client.indices.exists(index=self.index_name):
            mapping = {
                "mappings": {
                    "properties": {
                        "content": {"type": "text"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": 1536,  # Adjust based on your model
                            "index": True,
                            "similarity": "cosine"
                        },
                        "user_id": {"type": "keyword"},
                        "doc_id": {"type": "keyword"},
                        "ingested_at": {"type": "date"},
                        "source": {"type": "keyword"},
                        "title": {"type": "text"},
                        "document_type": {"type": "keyword"}
                    }
                }
            }
            self.client.indices.create(index=self.index_name, body=mapping)
    
    def add_documents(
        self, 
        documents: List[Document], 
        user_id: str,
        batch_size: int = 100,
        **kwargs
    ) -> List[str]:
        """Add documents to Elasticsearch"""
        
        # Prepare documents with user ID
        prepared_docs = self._prepare_documents(documents, user_id)
        
        # Get embeddings
        texts = [doc.page_content for doc in prepared_docs]
        embeddings = self.embedding_model.embed_documents(texts)
        
        doc_ids = []
        
        # Bulk insert
        bulk_data = []
        for doc, embedding in zip(prepared_docs, embeddings):
            doc_id = doc.metadata["doc_id"]
            doc_ids.append(doc_id)
            
            # Create document
            document = {
                "content": doc.page_content,
                "embedding": embedding,
                **doc.metadata
            }
            
            bulk_data.append({"index": {"_index": self.index_name, "_id": doc_id}})
            bulk_data.append(document)
        
        # Execute bulk insert
        if bulk_data:
            response = self.client.bulk(operations=bulk_data, refresh=True)
            
            # Check for errors
            if response.get("errors"):
                for item in response["items"]:
                    if "error" in item["index"]:
                        print(f"Error indexing document: {item['index']['error']}")
        
        return doc_ids
    
    def search(
        self, 
        query: str, 
        user_id: str, 
        k: int = 5,
        filters: Optional[Dict] = None,
        **kwargs
    ) -> List[Document]:
        """Search documents for a specific user"""
        
        # Get query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Build query
        search_query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                    "params": {"query_vector": query_embedding}
                                }
                            }
                        }
                    ],
                    "filter": [
                        {"term": {"user_id": user_id}}
                    ]
                }
            },
            "size": k
        }
        
        # Add additional filters
        if filters:
            for key, value in filters.items():
                search_query["query"]["bool"]["filter"].append({"term": {key: value}})
        
        # Execute search
        response = self.client.search(index=self.index_name, body=search_query)
        
        # Convert to LangChain Documents
        documents = []
        for hit in response["hits"]["hits"]:
            doc = Document(
                page_content=hit["_source"]["content"],
                metadata={
                    **hit["_source"],
                    "score": hit["_score"]
                }
            )
            documents.append(doc)
        
        return documents
    
    def delete_user_documents(self, user_id: str) -> bool:
        """Delete all documents for a user"""
        try:
            response = self.client.delete_by_query(
                index=self.index_name,
                body={"query": {"term": {"user_id": user_id}}}
            )
            return response["deleted"] > 0
        except Exception as e:
            print(f"Error deleting user documents: {e}")
            return False
    
    def get_user_document_count(self, user_id: str) -> int:
        """Get document count for a user"""
        try:
            response = self.client.count(
                index=self.index_name,
                body={"query": {"term": {"user_id": user_id}}}
            )
            return response["count"]
        except Exception as e:
            print(f"Error getting document count: {e}")
            return 0