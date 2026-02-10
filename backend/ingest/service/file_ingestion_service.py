"""Load files from various formats, clean up, split, ingest into Weaviate."""

import logging
import os
import re
import uuid
from typing import Optional, List, Dict, Any
from pathlib import Path

import weaviate
from bs4 import BeautifulSoup
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    CSVLoader
)
from langchain_community.indexes._sql_record_manager import SQLRecordManager
from langchain_core.indexing import index
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate import WeaviateVectorStore

from backend.agent.constants import WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX
from backend.agent.embeddings import get_embeddings_model
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
RECORD_MANAGER_DB_URL = os.environ["RECORD_MANAGER_DB_URL"]

# Supported file extensions and their corresponding loaders
SUPPORTED_EXTENSIONS = {
    '.pdf': PyPDFLoader,
    '.docx': Docx2txtLoader,
    '.doc': Docx2txtLoader,
    '.txt': TextLoader,
    '.xlsx': UnstructuredExcelLoader,
    '.xls': UnstructuredExcelLoader,
    '.csv': CSVLoader,
    '.pptx': UnstructuredPowerPointLoader,
    '.ppt': UnstructuredPowerPointLoader,
}

def clean_metadata_key(key: str) -> str:
    """Clean a single metadata key to be Weaviate-compatible."""
    # Replace dots, hyphens, and other invalid chars with underscores
    clean_key = re.sub(r'[^_A-Za-z0-9]', '_', key)
    
    # Ensure it starts with letter or underscore (GraphQL requirement)
    if not re.match(r'^[_A-Za-z]', clean_key):
        clean_key = '_' + clean_key
    
    # Truncate if too long (Weaviate limit is 231 chars)
    if len(clean_key) > 230:
        clean_key = clean_key[:230]
    
    return clean_key

def clean_metadata(metadata: dict) -> dict:
    """Clean all metadata keys in a dictionary."""
    cleaned = {}
    for key, value in metadata.items():
        clean_key = clean_metadata_key(key)
        cleaned[clean_key] = value
    return cleaned

def load_documents_from_files(files: List[str], user_id: str = None) -> List[Document]:
    """Load documents from various file formats with Weaviate-compatible metadata.
    
    Args:
        files: List of file paths to load
        user_id: Optional user ID to associate with documents
    """
    docs = []
    
    for file_path in files:
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext not in SUPPORTED_EXTENSIONS:
            logger.warning(f"Unsupported file type: {file_ext} for file {file_path}")
            continue
            
        try:
            # Get appropriate loader for file type
            loader_class = SUPPORTED_EXTENSIONS[file_ext]
            
            # Different loaders have different initialization parameters
            if file_ext in ['.pdf', '.docx', '.doc', '.txt']:
                loader = loader_class(file_path)
            elif file_ext in ['.xlsx', '.xls', '.csv']:
                loader = loader_class(file_path, mode="elements")
            elif file_ext in ['.pptx', '.ppt']:
                loader = loader_class(file_path, mode="elements")
            else:
                loader = loader_class(file_path)
            
            # Load documents
            loaded_docs = loader.load()
            
            # Clean metadata for each document
            for doc in loaded_docs:
                cleaned_metadata = clean_metadata(doc.metadata)
                
                # Add file metadata
                cleaned_metadata['file_name'] = Path(file_path).name
                cleaned_metadata['file_path'] = str(Path(file_path).absolute())
                cleaned_metadata['file_extension'] = file_ext
                cleaned_metadata['file_size'] = os.path.getsize(file_path)
                
                # Add file type category
                if file_ext in ['.pdf']:
                    cleaned_metadata['file_type'] = 'pdf'
                elif file_ext in ['.docx', '.doc']:
                    cleaned_metadata['file_type'] = 'word'
                elif file_ext in ['.xlsx', '.xls', '.csv']:
                    cleaned_metadata['file_type'] = 'excel'
                elif file_ext in ['.pptx', '.ppt']:
                    cleaned_metadata['file_type'] = 'powerpoint'
                elif file_ext in ['.txt']:
                    cleaned_metadata['file_type'] = 'text'
                
                # Add user_id to metadata if provided
                if user_id:
                    cleaned_metadata['user_id'] = str(user_id)
                
                # Ensure source path uses forward slashes
                if 'source' in cleaned_metadata:
                    cleaned_metadata['source'] = str(cleaned_metadata['source']).replace('\\', '/')
                else:
                    cleaned_metadata['source'] = str(Path(file_path).absolute()).replace('\\', '/')
                
                # Add title if not present
                if 'title' not in cleaned_metadata:
                    cleaned_metadata['title'] = Path(file_path).stem
                
                # Add unique ID for each chunk
                cleaned_metadata['chunk_id'] = str(uuid.uuid4())
                
                # Create new document with cleaned metadata
                clean_doc = Document(
                    page_content=doc.page_content,
                    metadata=cleaned_metadata
                )
                docs.append(clean_doc)
                
            logger.info(f"Successfully loaded {len(loaded_docs)} documents from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            continue
    
    logger.info(f"Loaded {len(docs)} total documents from {len(files)} files")
    return docs

def load_documents_from_directory(directory: str, user_id: str = None) -> List[Document]:
    """Load all supported documents from a directory.
    
    Args:
        directory: Path to directory
        user_id: Optional user ID to associate with documents
    """
    files_to_process = []
    
    for ext in SUPPORTED_EXTENSIONS.keys():
        files_to_process.extend(Path(directory).glob(f"**/*{ext}"))
        files_to_process.extend(Path(directory).glob(f"**/*{ext.upper()}"))
    
    return load_documents_from_files([str(f) for f in files_to_process], user_id)

def ingest_documents(
    files: List[str] = None, 
    directory: str = None, 
    user_id: str = None,
    namespace: str = None
) -> Dict[str, Any]:
    """Main ingestion function that handles multiple file formats.
    
    Args:
        files: List of file paths to ingest
        directory: Directory path to ingest all files from
        user_id: User ID to associate with documents
        namespace: Optional namespace for multi-tenancy (used in source_id_key)
    """
    
    # Get documents from files or directory
    if files:
        docs = load_documents_from_files(files, user_id)
    elif directory and os.path.exists(directory):
        docs = load_documents_from_directory(directory, user_id)
    else:
        # Default path
        default_path = "../assets/pdfs"
        if os.path.exists(default_path):
            docs = load_documents_from_directory(default_path, user_id)
        else:
            logger.info("No documents found to ingest.")
            return {"status": "no_docs", "total_vectors": 0}
    
    if not docs:
        logger.info("No documents found to ingest.")
        return {"status": "no_docs", "total_vectors": 0}
    
    # Setup text splitter, embedding model, and vector store
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    embedding = get_embeddings_model()
    
    with weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=weaviate.classes.init.Auth.api_key(WEAVIATE_API_KEY),
        skip_init_checks=True,
    ) as weaviate_client:
        # Setup vector store with additional attributes
        vectorstore = WeaviateVectorStore(
            client=weaviate_client,
            index_name=WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX,
            text_key="text",
            embedding=embedding,
            attributes=["source", "title", "file_type", "file_name", "user_id", "chunk_id"],
        )
        
        # Setup record manager
        record_manager = SQLRecordManager(
            f"weaviate/{WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX}",
            db_url=RECORD_MANAGER_DB_URL,
        )
        record_manager.create_schema()
        
        # Split documents
        docs_transformed = text_splitter.split_documents(docs)
        docs_transformed = [
            doc for doc in docs_transformed if len(doc.page_content) > 10
        ]
        
        # Ensure required metadata fields are present
        for doc in docs_transformed:
            if "source" not in doc.metadata:
                doc.metadata["source"] = ""
            if "title" not in doc.metadata:
                doc.metadata["title"] = ""
            if "file_type" not in doc.metadata:
                doc.metadata["file_type"] = "unknown"
            if "file_name" not in doc.metadata:
                doc.metadata["file_name"] = ""
            if "user_id" not in doc.metadata:
                doc.metadata["user_id"] = user_id or "anonymous"
            if "chunk_id" not in doc.metadata:
                doc.metadata["chunk_id"] = str(uuid.uuid4())
        
        # Prepare source_id_key with namespace if provided
        source_id_key = "source"
        if namespace:
            # Add namespace to source for isolation
            for doc in docs_transformed:
                doc.metadata["source"] = f"{namespace}:{doc.metadata.get('source', '')}"
        
        # Index documents
        indexing_stats = index(
            docs_transformed,
            record_manager,
            vectorstore,
            cleanup="full",
            source_id_key=source_id_key,
            force_update=(os.environ.get("FORCE_UPDATE") or "false").lower() == "true",
        )
        
        # Get total vector count (with optional user filter)
        collection = weaviate_client.collections.get(
            WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX
        )
        
        # Count total vectors and user-specific vectors
        total_vecs = collection.aggregate.over_all().total_count
        
        # If user_id is provided, count vectors for that user
        user_vecs = 0
        if user_id:
            try:
                # Use aggregate with filter for user-specific count
                response = collection.aggregate.over_all(
                    filters=weaviate.classes.query.Filter.by_property("user_id").equal(user_id)
                )
                user_vecs = response.total_count
            except Exception as agg_error:
                logger.warning(f"Could not aggregate user vectors: {agg_error}")
                # Fallback - this won't give accurate count but indicates existence
                try:
                    response = collection.query.fetch_objects(
                        filters=weaviate.classes.query.Filter.by_property("user_id").equal(user_id),
                        limit=1
                    )
                    user_vecs = len(response.objects)  # This is 0 or 1, not total count
                except Exception as query_error:
                    logger.error(f"Could not query user vectors: {query_error}")
                    user_vecs = 0
                
                logger.info(f"Indexing stats: {indexing_stats}")
                logger.info(f"Total vectors in index: {total_vecs}")
                if user_id:
                    logger.info(f"Vectors for user {user_id}: {user_vecs}")
        
        return {
            "status": "success",
            "indexing_stats": indexing_stats,
            "total_vectors": total_vecs,
            "user_vectors": user_vecs if user_id else total_vecs,
            "documents_processed": len(docs),
            "chunks_created": len(docs_transformed),
            "user_id": user_id,
            "namespace": namespace,
        }

if __name__ == "__main__":
    # For backward compatibility
    ingest_documents()