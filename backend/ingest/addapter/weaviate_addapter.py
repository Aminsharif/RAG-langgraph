import weaviate
from weaviate.auth import AuthApiKey
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from typing import List, Dict, Any, Optional
from backend.ingest.addapter.base_vector_store import BaseVectorStore


class WeaviateVectorStore(BaseVectorStore):
    """Weaviate vector store implementation"""
    
    def __init__(
        self, 
        embedding_model: Embeddings,
        cluster_url: str,
        api_key: str,
        index_name: str = "Documents",
        text_key: str = "content"
    ):
        super().__init__(embedding_model)
        self.cluster_url = cluster_url
        self.api_key = api_key
        self.index_name = index_name
        self.text_key = text_key
        self._client = None
        
    @property
    def client(self):
        """Lazy client initialization"""
        if self._client is None:
            self._client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self.cluster_url,
                auth_credentials=AuthApiKey(self.api_key),
            )
        return self._client
    
    def add_documents(
        self, 
        documents: List[Document], 
        user_id: str,
        batch_size: int = 100,
        **kwargs
    ) -> List[str]:
        """Add documents to Weaviate"""
        
        # Prepare documents with user ID
        prepared_docs = self._prepare_documents(documents, user_id)
        
        # Get embeddings
        texts = [doc.page_content for doc in prepared_docs]
        embeddings = self.embedding_model.embed_documents(texts)
        
        doc_ids = []
        
        with self.client as client:
            # Get or create collection
            collection = client.collections.get_or_create(
                name=self.index_name,
                vectorizer_config=weaviate.classes.config.Configure.Vectorizer.none(),
            )
            
            # Add documents in batches
            for i in range(0, len(prepared_docs), batch_size):
                batch_docs = prepared_docs[i:i+batch_size]
                batch_embeddings = embeddings[i:i+batch_size]
                
                objects = []
                for doc, embedding in zip(batch_docs, batch_embeddings):
                    obj = weaviate.classes.data.DataObject(
                        properties={
                            self.text_key: doc.page_content,
                            **doc.metadata
                        },
                        vector=embedding
                    )
                    objects.append(obj)
                
                # Insert batch
                result = collection.data.insert_many(objects)
                doc_ids.extend([str(uuid) for uuid in result.uuids])
        
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
        
        # Build filter for user_id
        user_filter = weaviate.classes.query.Filter.by_property("user_id").equal(user_id)
        
        if filters:
            # Combine filters
            all_filters = weaviate.classes.query.Filter.and_(
                user_filter,
                weaviate.classes.query.Filter.by_property(filters["property"]).equal(filters["value"])
            )
        else:
            all_filters = user_filter
        
        with self.client as client:
            collection = client.collections.get(self.index_name)
            
            # Get query embedding
            query_embedding = self.embedding_model.embed_query(query)
            
            # Search
            response = collection.query.near_vector(
                near_vector=query_embedding,
                limit=k,
                filters=all_filters,
                return_metadata=weaviate.classes.query.MetadataQuery(distance=True)
            )
            
            # Convert to LangChain Documents
            documents = []
            for obj in response.objects:
                doc = Document(
                    page_content=obj.properties.get(self.text_key, ""),
                    metadata={
                        **obj.properties,
                        "distance": obj.metadata.distance
                    }
                )
                documents.append(doc)
        
        return documents
    
    def delete_user_documents(self, user_id: str) -> bool:
        """Delete all documents for a user"""
        try:
            with self.client as client:
                collection = client.collections.get(self.index_name)
                
                # Delete by user_id filter
                collection.data.delete_many(
                    where=weaviate.classes.query.Filter.by_property("user_id").equal(user_id)
                )
            return True
        except Exception as e:
            print(f"Error deleting user documents: {e}")
            return False
    
    def get_user_document_count(self, user_id: str) -> int:
        """Get document count for a user"""
        try:
            with self.client as client:
                collection = client.collections.get(self.index_name)
                
                response = collection.aggregate.over_all(
                    filters=weaviate.classes.query.Filter.by_property("user_id").equal(user_id)
                )
                return response.total_count
        except Exception as e:
            print(f"Error getting document count: {e}")
            return 0
    
    def close(self):
        """Close client connection"""
        if self._client:
            self._client.close() 