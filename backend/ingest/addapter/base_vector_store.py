from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
import uuid
import hashlib
from datetime import datetime


class BaseVectorStore(ABC):
    """Abstract base class for all vector stores"""
    
    def __init__(self, embedding_model: Embeddings):
        self.embedding_model = embedding_model
        
    def _create_document_id(self, text: str, user_id: str, source: str) -> str:
        """Create a unique document ID"""
        content_hash = hashlib.md5(text.encode()).hexdigest()[:16]
        return f"{user_id}_{source}_{content_hash}"
    
    def _prepare_documents(
        self, 
        documents: List[Document], 
        user_id: str,
        metadata_fields: Optional[List[str]] = None
    ) -> List[Document]:
        """Prepare documents with user ID and metadata"""
        prepared_docs = []
        
        for doc in documents:
            # Create document ID
            doc_id = self._create_document_id(
                doc.page_content, 
                user_id, 
                doc.metadata.get("source", "unknown")
            )
            
            # Add user ID and timestamp to metadata
            doc.metadata.update({
                "user_id": user_id,
                "doc_id": doc_id,
                "ingested_at": datetime.utcnow().isoformat(),
                "document_type": doc.metadata.get("document_type", "general")
            })
            
            # Keep only specified metadata fields if provided
            if metadata_fields:
                doc.metadata = {
                    k: v for k, v in doc.metadata.items() 
                    if k in metadata_fields + ["user_id", "doc_id", "ingested_at"]
                }
            
            prepared_docs.append(doc)
        
        return prepared_docs
    
    @abstractmethod
    def add_documents(self, documents: List[Document], user_id: str, **kwargs) -> List[str]:
        """Add documents to vector store"""
        pass
    
    @abstractmethod
    def search(self, query: str, user_id: str, k: int = 5, **kwargs) -> List[Document]:
        """Search documents for a specific user"""
        pass
    
    @abstractmethod
    def delete_user_documents(self, user_id: str) -> bool:
        """Delete all documents for a user"""
        pass
    
    @abstractmethod
    def get_user_document_count(self, user_id: str) -> int:
        """Get document count for a user"""
        pass