import pinecone
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from typing import List, Dict, Any, Optional
from backend.ingest.addapter.base_vector_store import BaseVectorStore


class PineconeVectorStore(BaseVectorStore):
    """Pinecone vector store implementation"""
    
    def __init__(
        self, 
        embedding_model: Embeddings,
        api_key: str,
        environment: str,
        index_name: str = "documents",
        namespace: str = "default"
    ):
        super().__init__(embedding_model)
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.namespace = namespace
        
        # Initialize Pinecone
        pinecone.init(api_key=api_key, environment=environment)
        
        # Get or create index
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=1536,  # Adjust based on your embedding model
                metric="cosine"
            )
        
        self.index = pinecone.Index(index_name)
    
    def add_documents(
        self, 
        documents: List[Document], 
        user_id: str,
        batch_size: int = 100,
        **kwargs
    ) -> List[str]:
        """Add documents to Pinecone"""
        
        # Prepare documents with user ID
        prepared_docs = self._prepare_documents(documents, user_id)
        
        # Get embeddings
        texts = [doc.page_content for doc in prepared_docs]
        embeddings = self.embedding_model.embed_documents(texts)
        
        doc_ids = []
        vectors = []
        
        for doc, embedding in zip(prepared_docs, embeddings):
            doc_id = doc.metadata["doc_id"]
            doc_ids.append(doc_id)
            
            # Prepare vector with metadata
            vector = {
                "id": doc_id,
                "values": embedding,
                "metadata": {
                    **doc.metadata,
                    "text": doc.page_content
                }
            }
            vectors.append(vector)
        
        # Upload in batches
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            self.index.upsert(
                vectors=batch,
                namespace=self.namespace
            )
        
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
        
        # Build filter
        filter_dict = {"user_id": {"$eq": user_id}}
        if filters:
            filter_dict.update(filters)
        
        # Get query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Search
        response = self.index.query(
            vector=query_embedding,
            top_k=k,
            namespace=self.namespace,
            filter=filter_dict,
            include_metadata=True
        )
        
        # Convert to LangChain Documents
        documents = []
        for match in response.matches:
            doc = Document(
                page_content=match.metadata.get("text", ""),
                metadata={
                    **match.metadata,
                    "score": match.score
                }
            )
            documents.append(doc)
        
        return documents
    
    def delete_user_documents(self, user_id: str) -> bool:
        """Delete all documents for a user"""
        try:
            self.index.delete(
                filter={"user_id": {"$eq": user_id}},
                namespace=self.namespace
            )
            return True
        except Exception as e:
            print(f"Error deleting user documents: {e}")
            return False
    
    def get_user_document_count(self, user_id: str) -> int:
        """Get document count for a user - Pinecone doesn't have direct count API"""
        # You might need to implement this differently for Pinecone
        try:
            # Sample query to get approximate count
            response = self.index.query(
                vector=[0] * 1536,  # Zero vector
                top_k=1,
                filter={"user_id": {"$eq": user_id}},
                include_metadata=False
            )
            # This doesn't give exact count, consider using describe_index_stats
            stats = self.index.describe_index_stats()
            # Pinecone stats are namespace-level, not user-level
            return 0  # Placeholder
        except Exception as e:
            print(f"Error getting document count: {e}")
            return 0