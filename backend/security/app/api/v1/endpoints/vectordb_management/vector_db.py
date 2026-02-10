# api.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uuid
from datetime import datetime
from pydantic import Field

from backend.ingest.config import WebsiteConfig, IndexingStrategy
from backend.ingest.service.ingestion_service import ingestion_service
from fastapi import File, UploadFile, HTTPException, Query, Depends, Header
from pydantic import BaseModel, Field
from pathlib import Path
from backend.ingest.service.file_ingestion_service import ingest_documents, SUPPORTED_EXTENSIONS
import os
import tempfile
import shutil
import logging

router = APIRouter()
# Store job status
ingestion_jobs: Dict[str, Dict[str, Any]] = {}

logger = logging.getLogger(__name__)

class IngestionRequest(BaseModel):
    url: str
    user_id: str
    name: Optional[str] = None
    index_name: Optional[str] = None
    strategy: Optional[str] = "sitemap"
    filter_urls: Optional[List[str]] = None
    allowed_domains: Optional[List[str]] = None
    max_depth: Optional[int] = 2
    chunk_size: Optional[int] = 4000
    chunk_overlap: Optional[int] = 200
    force_update: Optional[bool] = False

class IngestionResponse(BaseModel):
    job_id: str
    status: str
    message: str
    collection_name: Optional[str] = None

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime
class IngestionResponse(BaseModel):
    """Response model for ingestion operations."""
    status: str
    message: str
    job_id: Optional[str] = None
    total_vectors: Optional[int] = None
    user_vectors: Optional[int] = None
    documents_processed: Optional[int] = None
    chunks_created: Optional[int] = None
    indexing_stats: Optional[dict] = None
    user_id: Optional[str] = None

class BatchIngestionRequest(BaseModel):
    """Request model for batch ingestion."""
    file_paths: List[str] = Field(..., description="List of file paths to ingest")
    user_id: Optional[str] = Field(None, description="User ID for document isolation")
    namespace: Optional[str] = Field(None, description="Namespace for multi-tenancy")

class UserIngestionRequest(BaseModel):
    """Request model for user-specific ingestion."""
    user_id: str = Field(..., description="User ID for document isolation")
    namespace: Optional[str] = Field(None, description="Namespace for multi-tenancy")


def process_ingestion_job(job_id: str, config: WebsiteConfig, force_update: bool):
    """Background task to process ingestion"""
    try:
        ingestion_jobs[job_id]["status"] = "processing"
        ingestion_jobs[job_id]["updated_at"] = datetime.now()
        
        result = ingestion_service.ingest_website(config, force_update)
        ingestion_jobs[job_id].update({
            "status": "completed" if result["success"] else "failed",
            "result": result,
            "updated_at": datetime.now()
        })
    except Exception as e:
        ingestion_jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "updated_at": datetime.now()
        })

@router.post("/ingest", response_model=IngestionResponse)
async def ingest_website(
    request: IngestionRequest,
    background_tasks: BackgroundTasks
):
    """Start website ingestion"""
    try:
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Create website name if not provided
        name = request.name or f"Website_{job_id[:8]}"
        
        # Create config
        config = WebsiteConfig(
            url=request.url,
            user_id=request.user_id,
            name=name,
            index_name=request.index_name,
            strategy=IndexingStrategy(request.strategy),
            filter_urls=request.filter_urls,
            allowed_domains=request.allowed_domains,
            max_depth=request.max_depth,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
        
        # Store job
        ingestion_jobs[job_id] = {
            "status": "pending",
            "config": config.dict(),
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        
        # Start background task
        background_tasks.add_task(
            process_ingestion_job,
            job_id,
            config,
            request.force_update
        )
        
        return IngestionResponse(
            job_id=job_id,
            status="pending",
            message="Ingestion job started",
            collection_name=config.index_name
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Check ingestion job status"""
    if job_id not in ingestion_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = ingestion_jobs[job_id]
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        result=job.get("result"),
        error=job.get("error"),
        created_at=job["created_at"],
        updated_at=job["updated_at"]
    )

@router.get("/jobs")
async def list_jobs():
    """List all ingestion jobs"""
    return {
        "jobs": [
            {
                "job_id": job_id,
                "status": job["status"],
                "url": job["config"]["url"],
                "created_at": job["created_at"],
                "updated_at": job["updated_at"]
            }
            for job_id, job in ingestion_jobs.items()
        ]
    }

@router.get("/collections")
async def list_collections():
    """List all collections in ChromaDB"""
    collections = ingestion_service.client.list_collections()
    return {
        "collections": [
            {
                "name": collection.name,
                "metadata": collection.metadata,
                "count": collection.count()
            }
            for collection in collections
        ]
    }

@router.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """Delete a collection"""
    try:
        ingestion_service.client.delete_collection(collection_name)
        return {"success": True, "message": f"Collection {collection_name} deleted"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    



    
@router.post("/ingest/single", response_model=IngestionResponse)
async def ingest_single_file(
    file: UploadFile = File(...),
    user_id: str = Query(..., description="User ID for document isolation"),
    namespace: Optional[str] = Query(None, description="Namespace for multi-tenancy"),
    cleanup: bool = Query(default=True, description="Cleanup old vectors"),
    # Uncomment to use authentication:
    # current_user: str = Depends(validate_user),
):
    """Ingest a single file with user isolation."""
    
    # If using authentication, you can verify user_id matches current_user
    # if user_id != current_user:
    #     raise HTTPException(status_code=403, detail="Not authorized for this user")
    
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Supported types: {list(SUPPORTED_EXTENSIONS.keys())}"
        )
    
    # Create temporary directory for uploaded file
    temp_dir = tempfile.mkdtemp()
    try:
        # Save uploaded file to temp directory
        temp_file_path = os.path.join(temp_dir, file.filename)
        
        # Write uploaded file to temp location
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Ingest the file with user_id
        result = ingest_documents(
            files=[temp_file_path],
            user_id=user_id,
            namespace=namespace
        )
        
        if result["status"] == "success":
            return IngestionResponse(
                status="success",
                message=f"Successfully ingested {file.filename} for user {user_id}",
                total_vectors=result["total_vectors"],
                user_vectors=result["user_vectors"],
                documents_processed=result["documents_processed"],
                chunks_created=result["chunks_created"],
                indexing_stats=result["indexing_stats"],
                user_id=user_id
            )
        else:
            return IngestionResponse(
                status="error",
                message=f"Failed to ingest {file.filename} for user {user_id}: No documents found",
                user_id=user_id
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

@router.post("/ingest/multiple", response_model=IngestionResponse)
async def ingest_multiple_files(
    files: List[UploadFile] = File(...),
    user_id: str = Query(..., description="User ID for document isolation"),
    namespace: Optional[str] = Query(None, description="Namespace for multi-tenancy"),
    cleanup: bool = Query(default=True, description="Cleanup old vectors"),
    # current_user: str = Depends(validate_user),
):
    """Ingest multiple files at once with user isolation."""
    
    # if user_id != current_user:
    #     raise HTTPException(status_code=403, detail="Not authorized for this user")
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    temp_dir = tempfile.mkdtemp()
    temp_files = []
    
    try:
        # Save all uploaded files to temp directory
        for file in files:
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in SUPPORTED_EXTENSIONS:
                logger.warning(f"Skipping unsupported file type: {file_ext} - {file.filename}")
                continue
            
            temp_file_path = os.path.join(temp_dir, file.filename)
            with open(temp_file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            temp_files.append(temp_file_path)
        
        if not temp_files:
            raise HTTPException(
                status_code=400,
                detail="No valid files to process. Supported types: " + str(list(SUPPORTED_EXTENSIONS.keys()))
            )
        
        # Ingest all files with user_id
        result = ingest_documents(
            files=temp_files,
            user_id=user_id,
            namespace=namespace
        )
        
        if result["status"] == "success":
            return IngestionResponse(
                status="success",
                message=f"Successfully ingested {len(temp_files)} files for user {user_id}",
                total_vectors=result["total_vectors"],
                user_vectors=result["user_vectors"],
                documents_processed=result["documents_processed"],
                chunks_created=result["chunks_created"],
                indexing_stats=result["indexing_stats"],
                user_id=user_id
            )
        else:
            return IngestionResponse(
                status="error",
                message=f"Failed to ingest files for user {user_id}: No documents found",
                user_id=user_id
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")
    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

@router.post("/ingest/batch", response_model=IngestionResponse)
async def ingest_batch_files(
    request: BatchIngestionRequest,
    # current_user: str = Depends(validate_user),
):
    """Ingest multiple files from file paths with user isolation."""
    
    user_id = request.user_id
    # if user_id and user_id != current_user:
    #     raise HTTPException(status_code=403, detail="Not authorized for this user")
    
    if not request.file_paths:
        raise HTTPException(status_code=400, detail="No file paths provided")
    
    # Validate file paths
    valid_files = []
    for file_path in request.file_paths:
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            continue
        
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in SUPPORTED_EXTENSIONS:
            logger.warning(f"Unsupported file type: {file_ext} - {file_path}")
            continue
        
        valid_files.append(file_path)
    
    if not valid_files:
        raise HTTPException(
            status_code=400,
            detail="No valid files found to process"
        )
    
    # Ingest files with user_id
    result = ingest_documents(
        files=valid_files,
        user_id=user_id,
        namespace=request.namespace
    )
    
    if result["status"] == "success":
        return IngestionResponse(
            status="success",
            message=f"Successfully ingested {len(valid_files)} files for user {user_id}",
            total_vectors=result["total_vectors"],
            user_vectors=result["user_vectors"],
            documents_processed=result["documents_processed"],
            chunks_created=result["chunks_created"],
            indexing_stats=result["indexing_stats"],
            user_id=user_id
        )
    else:
        return IngestionResponse(
            status="error",
            message=f"Failed to ingest files for user {user_id}",
            user_id=user_id
        )

@router.post("/ingest/directory", response_model=IngestionResponse)
async def ingest_directory(
    directory_path: str = Query(..., description="Path to directory containing documents"),
    user_id: str = Query(..., description="User ID for document isolation"),
    namespace: Optional[str] = Query(None, description="Namespace for multi-tenancy"),
    cleanup: bool = Query(default=True, description="Cleanup old vectors"),
    # current_user: str = Depends(validate_user),
):
    """Ingest all supported documents from a directory with user isolation."""
    
    # if user_id != current_user:
    #     raise HTTPException(status_code=403, detail="Not authorized for this user")
    
    if not os.path.exists(directory_path):
        raise HTTPException(status_code=400, detail="Directory not found")
    
    if not os.path.isdir(directory_path):
        raise HTTPException(status_code=400, detail="Path is not a directory")
    
    try:
        result = ingest_documents(
            directory=directory_path,
            user_id=user_id,
            namespace=namespace
        )
        
        if result["status"] == "success":
            return IngestionResponse(
                status="success",
                message=f"Successfully ingested documents from {directory_path} for user {user_id}",
                total_vectors=result["total_vectors"],
                user_vectors=result["user_vectors"],
                documents_processed=result["documents_processed"],
                chunks_created=result["chunks_created"],
                indexing_stats=result["indexing_stats"],
                user_id=user_id
            )
        else:
            return IngestionResponse(
                status="error",
                message=f"No documents found in directory for user {user_id}",
                user_id=user_id
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing directory: {str(e)}")

@router.post("/ingest/user", response_model=IngestionResponse)
async def setup_user_namespace(
    request: UserIngestionRequest,
    # current_user: str = Depends(validate_user),
):
    """Setup user namespace for document ingestion."""
    
    user_id = request.user_id
    # if user_id != current_user:
    #     raise HTTPException(status_code=403, detail="Not authorized for this user")
    
    # This endpoint can be used to initialize user-specific settings
    # For now, just return success
    return IngestionResponse(
        status="success",
        message=f"User namespace setup complete for {user_id}",
        user_id=user_id,
        namespace=request.namespace
    )

@router.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported file formats."""
    return {
        "supported_formats": list(SUPPORTED_EXTENSIONS.keys()),
        "loaders": {ext: loader.__name__ for ext, loader in SUPPORTED_EXTENSIONS.items()}
    }

@router.get("/stats")
async def get_ingestion_stats(
    user_id: Optional[str] = Query(None, description="Filter stats by user ID"),
    # current_user: str = Depends(validate_user),
):
    """Get ingestion statistics with optional user filter."""
    
    # if user_id and user_id != current_user:
    #     raise HTTPException(status_code=403, detail="Not authorized to view this user's stats")
    
    try:
        import weaviate
        WEAVIATE_URL = os.environ["WEAVIATE_URL"]
        WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
        
        with weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=weaviate.classes.init.Auth.api_key(WEAVIATE_API_KEY),
            skip_init_checks=True,
        ) as client:
            from backend.agent.constants import WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX
            
            collection = client.collections.get(WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX)
            
            # Total vectors
            total_vecs = collection.aggregate.over_all().total_count
            
            # User-specific vectors if user_id provided
            user_vecs = None
            if user_id:
                # Note: This requires proper indexing of user_id property
                response = collection.query.fetch_objects(
                    filters=weaviate.classes.query.Filter.by_property("user_id").equal(user_id),
                    limit=1,
                    return_metadata=weaviate.classes.query.MetadataQuery(cursor=True)
                )
                user_vecs = len(response.objects)
            
            stats = {
                "total_vectors": total_vecs,
                "index_name": WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX,
                "user_filter_applied": user_id is not None,
            }
            
            if user_id:
                stats["user_id"] = user_id
                stats["user_vectors"] = user_vecs
            
            return stats
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@router.get("/user/documents")
async def get_user_documents(
    user_id: str = Query(..., description="User ID to get documents for"),
    limit: int = Query(10, description="Number of documents to return"),
    offset: int = Query(0, description="Offset for pagination"),
    # current_user: str = Depends(validate_user),
):
    """Get list of documents for a specific user."""
    
    # if user_id != current_user:
    #     raise HTTPException(status_code=403, detail="Not authorized to view this user's documents")
    
    try:
        import weaviate
        WEAVIATE_URL = os.environ["WEAVIATE_URL"]
        WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
        
        with weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=weaviate.classes.init.Auth.api_key(WEAVIATE_API_KEY),
            skip_init_checks=True,
        ) as client:
            from backend.agent.constants import WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX
            
            collection = client.collections.get(WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX)
            
            # Query documents for specific user
            response = collection.query.fetch_objects(
                filters=weaviate.classes.query.Filter.by_property("user_id").equal(user_id),
                limit=limit,
                offset=offset,
                return_properties=["source", "title", "file_name", "file_type", "chunk_id"]
            )
            
            documents = []
            for obj in response.objects:
                doc = {
                    "id": str(obj.uuid),
                    "chunk_id": obj.properties.get("chunk_id"),
                    "title": obj.properties.get("title"),
                    "file_name": obj.properties.get("file_name"),
                    "file_type": obj.properties.get("file_type"),
                    "source": obj.properties.get("source"),
                }
                documents.append(doc)
            
            return {
                "user_id": user_id,
                "documents": documents,
                "total_returned": len(documents),
                "limit": limit,
                "offset": offset,
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting user documents: {str(e)}")

