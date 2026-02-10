import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from typing import List, Dict, Any, Optional
from backend.ingest.addapter.base_vector_store import BaseVectorStore

class ChromaVectorStore(BaseVectorStore):
    """ChromaDB vector store implementation"""
    
    def __init__(
        self, 
        embedding_model: Embeddings,
        persist_directory: str = "./chroma_db",
        collection_name: str = "documents"
    ):
        super().__init__(embedding_model)
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(
        self, 
        documents: List[Document], 
        user_id: str,
        batch_size: int = 100,
        **kwargs
    ) -> List[str]:
        """Add documents to Chroma"""
        
        # Prepare documents with user ID
        prepared_docs = self._prepare_documents(documents, user_id)
        
        # Get embeddings
        texts = [doc.page_content for doc in prepared_docs]
        embeddings = self.embedding_model.embed_documents(texts)
        
        # Generate IDs
        doc_ids = [doc.metadata["doc_id"] for doc in prepared_docs]
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=[doc.metadata for doc in prepared_docs],
            ids=doc_ids
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
        
        # Build where filter
        where_filter = {"user_id": user_id}
        if filters:
            where_filter.update(filters)
        
        # Get query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        # Convert to LangChain Documents
        documents = []
        for i in range(len(results["documents"][0])):
            doc = Document(
                page_content=results["documents"][0][i],
                metadata={
                    **results["metadatas"][0][i],
                    "distance": results["distances"][0][i]
                }
            )
            documents.append(doc)
        
        return documents
    
    def delete_user_documents(self, user_id: str) -> bool:
        """Delete all documents for a user"""
        try:
            self.collection.delete(where={"user_id": user_id})
            return True
        except Exception as e:
            print(f"Error deleting user documents: {e}")
            return False
    
    def get_user_document_count(self, user_id: str) -> int:
        """Get document count for a user"""
        try:
            count = self.collection.count(where={"user_id": user_id})
            return count
        except Exception as e:
            print(f"Error getting document count: {e}")
            return 0