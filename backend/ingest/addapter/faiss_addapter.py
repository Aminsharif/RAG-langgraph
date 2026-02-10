import faiss
import numpy as np
import pickle
from typing import List, Optional, Dict
import os
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from backend.ingest.addapter.base_vector_store import BaseVectorStore


class FAISSVectorStore(BaseVectorStore):
    """FAISS vector store implementation with user isolation"""
    
    def __init__(
        self, 
        embedding_model: Embeddings,
        index_path: str = "./faiss_index",
        dimension: int = 1536
    ):
        super().__init__(embedding_model)
        self.index_path = index_path
        self.dimension = dimension
        
        # Create directory if not exists
        os.makedirs(index_path, exist_ok=True)
        
        # Load or create index
        self.index = self._load_or_create_index()
        self.documents = []  # Store documents in memory
        self.metadatas = []  # Store metadata in memory
        
        # Load existing data if available
        self._load_data()
    
    def _load_or_create_index(self):
        """Load existing FAISS index or create new one"""
        index_file = os.path.join(self.index_path, "faiss.index")
        
        if os.path.exists(index_file):
            return faiss.read_index(index_file)
        else:
            return faiss.IndexFlatIP(self.dimension)  # Inner product (cosine similarity)
    
    def _load_data(self):
        """Load documents and metadata from disk"""
        docs_file = os.path.join(self.index_path, "documents.pkl")
        metas_file = os.path.join(self.index_path, "metadatas.pkl")
        
        if os.path.exists(docs_file):
            with open(docs_file, 'rb') as f:
                self.documents = pickle.load(f)
        
        if os.path.exists(metas_file):
            with open(metas_file, 'rb') as f:
                self.metadatas = pickle.load(f)
    
    def _save_data(self):
        """Save documents and metadata to disk"""
        docs_file = os.path.join(self.index_path, "documents.pkl")
        metas_file = os.path.join(self.index_path, "metadatas.pkl")
        
        with open(docs_file, 'wb') as f:
            pickle.dump(self.documents, f)
        
        with open(metas_file, 'wb') as f:
            pickle.dump(self.metadatas, f)
        
        # Save index
        index_file = os.path.join(self.index_path, "faiss.index")
        faiss.write_index(self.index, index_file)
    
    def add_documents(
        self, 
        documents: List[Document], 
        user_id: str,
        **kwargs
    ) -> List[str]:
        """Add documents to FAISS"""
        
        # Prepare documents with user ID
        prepared_docs = self._prepare_documents(documents, user_id)
        
        # Get embeddings
        texts = [doc.page_content for doc in prepared_docs]
        embeddings = self.embedding_model.embed_documents(texts)
        
        # Normalize vectors for cosine similarity
        embeddings_np = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings_np)
        
        # Add to index
        self.index.add(embeddings_np)
        
        # Store documents and metadata
        doc_ids = []
        for doc in prepared_docs:
            self.documents.append(doc.page_content)
            self.metadatas.append(doc.metadata)
            doc_ids.append(doc.metadata["doc_id"])
        
        # Save to disk
        self._save_data()
        
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
        query_embedding_np = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_embedding_np)
        
        # Search in index
        distances, indices = self.index.search(query_embedding_np, min(k * 10, len(self.documents)))
        
        # Filter by user_id and other criteria
        documents = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.metadatas):
                continue
            
            metadata = self.metadatas[idx]
            
            # Check user_id
            if metadata.get("user_id") != user_id:
                continue
            
            # Apply additional filters
            if filters:
                skip = False
                for key, value in filters.items():
                    if metadata.get(key) != value:
                        skip = True
                        break
                if skip:
                    continue
            
            # Create document
            doc = Document(
                page_content=self.documents[idx],
                metadata={
                    **metadata,
                    "distance": float(distances[0][i])
                }
            )
            documents.append(doc)
            
            if len(documents) >= k:
                break
        
        return documents
    
    def delete_user_documents(self, user_id: str) -> bool:
        """Delete all documents for a user"""
        try:
            # Find indices to remove
            indices_to_remove = [
                i for i, metadata in enumerate(self.metadatas)
                if metadata.get("user_id") == user_id
            ]
            
            if not indices_to_remove:
                return True
            
            # Remove from index (FAISS doesn't support deletion, so we need to rebuild)
            self._rebuild_index_without_user(user_id)
            return True
        except Exception as e:
            print(f"Error deleting user documents: {e}")
            return False
    
    def _rebuild_index_without_user(self, user_id: str):
        """Rebuild index without documents from specific user"""
        # Collect data to keep
        documents_to_keep = []
        metadatas_to_keep = []
        embeddings_to_keep = []
        
        for i, metadata in enumerate(self.metadatas):
            if metadata.get("user_id") != user_id:
                documents_to_keep.append(self.documents[i])
                metadatas_to_keep.append(metadata)
                # We need to get embeddings again or store them separately
                # For simplicity, we'll re-embed
                embedding = self.embedding_model.embed_documents([self.documents[i]])[0]
                embeddings_to_keep.append(embedding)
        
        # Create new index
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Add embeddings back
        if embeddings_to_keep:
            embeddings_np = np.array(embeddings_to_keep).astype('float32')
            faiss.normalize_L2(embeddings_np)
            self.index.add(embeddings_np)
        
        # Update stored data
        self.documents = documents_to_keep
        self.metadatas = metadatas_to_keep
        
        # Save
        self._save_data()
    
    def get_user_document_count(self, user_id: str) -> int:
        """Get document count for a user"""
        try:
            count = sum(1 for metadata in self.metadatas if metadata.get("user_id") == user_id)
            return count
        except Exception as e:
            print(f"Error getting document count: {e}")
            return 0