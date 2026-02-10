# ingestion_service.py
import logging
import os
from typing import Optional, Dict, Any
from urllib.parse import urlparse
from dotenv import load_dotenv
load_dotenv()

import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import SitemapLoader, RecursiveUrlLoader
from langchain_community.indexes._sql_record_manager import SQLRecordManager
from langchain_core.indexing import index
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup, SoupStrainer

from backend.ingest.config import WebsiteConfig, IndexingStrategy
from backend.ingest.parser import langchain_docs_extractor
from backend.agent.embeddings import get_embeddings_model
import weaviate
from langchain_weaviate import WeaviateVectorStore
from weaviate.classes.init import Auth
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError, HTTPError
from typing_extensions import List
from langchain_core.documents import Document
from backend.ingest.vector_store_factory import VectorStoreFactory, VectorStoreType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
RECORD_MANAGER_DB_URL = os.environ["RECORD_MANAGER_DB_URL"]

class IngestionService:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000, 
            chunk_overlap=200
        )
        self.embedding = get_embeddings_model()
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
    @staticmethod
    def metadata_extractor(meta: dict, soup: BeautifulSoup) -> dict:
        """Extract metadata from HTML soup"""
        title = soup.find("title")
        description = soup.find("meta", attrs={"name": "description"})
        html = soup.find("html")
        return {
            "source": meta.get("loc", ""),
            "title": title.get_text() if title else "",
            "description": description.get("content", "") if description else "",
            "language": html.get("lang", "") if html else "",
            **meta,
        }
    
    def _extract_domain_from_url(self, url: str) -> str:
        """Extract domain from URL for collection naming"""
        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "").replace(".", "_")
        return domain
    
    def _create_index_name(self, config: WebsiteConfig) -> str:
        """Create a unique index name based on config"""
        if config.index_name:
            return config.index_name
        
        domain = self._extract_domain_from_url(str(config.url))
        return f"{domain}_{config.strategy}"
    
    # def load_documents(self, config: WebsiteConfig):
    #     """Load documents based on indexing strategy"""
    #     if config.strategy == IndexingStrategy.SITEMAP:
    #         # Try sitemap first
    #         sitemap_urls = [
    #             f"{config.url}/sitemap.xml",
    #             f"{config.url}/sitemap_index.xml",
    #             f"{config.url}/sitemap",
    #         ]
            
    #         for sitemap_url in sitemap_urls:
    #             try:
    #                 loader = SitemapLoader(
    #                     sitemap_url,
    #                     filter_urls=config.filter_urls or [str(config.url)],
    #                     parsing_function=langchain_docs_extractor,
    #                     default_parser="lxml",
    #                     bs_kwargs={
    #                         "parse_only": SoupStrainer(
    #                             name=("article", "title", "html", "lang", "content")
    #                         ),
    #                     },
    #                     meta_function=self.metadata_extractor,
    #                 )
    #                 return loader.load()
    #             except Exception as e:
    #                 logger.warning(f"Failed to load sitemap from {sitemap_url}: {e}")
    #                 continue
            
    #         # Fall back to recursive if sitemap not found
    #         config.strategy = IndexingStrategy.RECURSIVE
        
    #     if config.strategy == IndexingStrategy.RECURSIVE:
    #         loader = RecursiveUrlLoader(
    #             url=str(config.url),
    #             max_depth=config.max_depth or 2,
    #             extractor=langchain_docs_extractor,
    #             prevent_outside=config.allowed_domains is None,
    #             allowed_domains=config.allowed_domains,
    #             use_async=True,
    #         )
    #         return loader.load()
        
    #     raise ValueError(f"Unsupported strategy: {config.strategy}")
    
    def load_documents(self, config: WebsiteConfig):
        """Load documents based on indexing strategy with robust error handling"""
    
        def check_url_exists(url: str, timeout: int = 5) -> bool:
            """Check if a URL exists and returns valid content"""
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (compatible; DocumentIngestionBot/1.0)'
                }
                response = requests.head(url, timeout=timeout, headers=headers, allow_redirects=True)
                
                # Check if it's a valid sitemap URL
                if response.status_code == 200:
                    # Try to get content to confirm it's a sitemap
                    content_response = requests.get(url, timeout=timeout, headers=headers)
                    content = content_response.text.lower()
                    
                    # Check if it looks like a sitemap
                    if 'xml' in content_response.headers.get('content-type', '').lower():
                        return True
                    # Check for common sitemap indicators in content
                    if any(indicator in content for indicator in ['<urlset', '<sitemapindex', 'xmlns="http://www.sitemaps.org']):
                        return True
                    
                return False
                
            except (RequestException, Timeout, ConnectionError, HTTPError) as e:
                logger.debug(f"URL {url} not accessible: {e}")
                return False
            except Exception as e:
                logger.debug(f"Error checking URL {url}: {e}")
                return False
        
        def try_sitemap_loader(sitemap_url: str) -> List[Document]:
            """Try to load documents from a sitemap URL"""
            try:
                logger.info(f"Attempting to load from sitemap: {sitemap_url}")
                
                loader = SitemapLoader(
                    sitemap_url,
                    filter_urls=config.filter_urls or [str(config.url)],
                    parsing_function=langchain_docs_extractor,
                    default_parser="lxml",
                    bs_kwargs={
                        "parse_only": SoupStrainer(
                            name=("article", "title", "html", "lang", "content", "main", "body")
                        ),
                    },
                    meta_function=self.metadata_extractor,
                    continue_on_failure=True,  # Continue if some URLs fail
                    requests_per_second=2,  # Be respectful to the server
                )
                
                docs = loader.load()
                if docs:
                    logger.info(f"Successfully loaded {len(docs)} documents from sitemap: {sitemap_url}")
                return docs
                
            except Exception as e:
                logger.warning(f"Failed to load documents from sitemap {sitemap_url}: {e}")
                return []
        
        def discover_sitemap_urls(base_url: str) -> List[str]:
            """Discover potential sitemap URLs"""
            parsed_url = urlparse(base_url)
            base_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
            
            # Common sitemap locations
            common_paths = [
                "/sitemap.xml",
                "/sitemap_index.xml",
                "/sitemap",
                "/sitemap.php",
                "/sitemap.txt",
                "/sitemap.xml.gz",
                "/sitemap1.xml",
                "/post-sitemap.xml",
                "/page-sitemap.xml",
                "/wp-sitemap.xml",
                "/robots.txt",  # Check robots.txt for sitemap location
            ]
            
            # Check robots.txt for sitemap directive
            robots_url = f"{base_domain}/robots.txt"
            sitemap_urls = []
            
            try:
                response = requests.get(robots_url, timeout=5)
                if response.status_code == 200:
                    lines = response.text.split('\n')
                    for line in lines:
                        if line.lower().startswith('sitemap:'):
                            sitemap_url = line.split(':', 1)[1].strip()
                            if sitemap_url:
                                sitemap_urls.append(sitemap_url)
                                logger.info(f"Found sitemap in robots.txt: {sitemap_url}")
            except Exception as e:
                logger.debug(f"Could not fetch robots.txt: {e}")
            
            # Add common sitemap paths
            for path in common_paths:
                sitemap_urls.append(f"{base_domain}{path}")
            
            # Also check the provided URL path
            if parsed_url.path and parsed_url.path != '/':
                base_path = parsed_url.path.rsplit('/', 1)[0]
                for path in common_paths:
                    sitemap_urls.append(f"{base_domain}{base_path}{path}")
            
            return list(set(sitemap_urls))  # Remove duplicates
        
        logger.info(f"Loading documents from {config.url} using {config.strategy} strategy")
        
        if config.strategy == IndexingStrategy.SITEMAP:
            # Discover potential sitemap URLs
            sitemap_urls = discover_sitemap_urls(str(config.url))
            logger.info(f"Discovered {len(sitemap_urls)} potential sitemap URLs")
            
            # Try each sitemap URL
            for sitemap_url in sitemap_urls:
                try:
                    # First check if URL exists
                    logger.debug(f"Checking sitemap URL: {sitemap_url}")
                    if check_url_exists(sitemap_url):
                        logger.info(f"Sitemap found at: {sitemap_url}")
                        
                        # Try to load from this sitemap
                        docs = try_sitemap_loader(sitemap_url)
                        if docs:
                            return docs
                        else:
                            logger.warning(f"Sitemap found but no documents loaded from: {sitemap_url}")
                    else:
                        logger.debug(f"Sitemap not found at: {sitemap_url}")
                        
                except Exception as e:
                    logger.warning(f"Error processing sitemap URL {sitemap_url}: {e}")
                    continue
            
            # If no sitemap found or no documents loaded, fall back to recursive
            logger.warning("No valid sitemap found or no documents loaded. Falling back to recursive strategy.")
            config.strategy = IndexingStrategy.RECURSIVE
        
        if config.strategy == IndexingStrategy.RECURSIVE:
            try:
                logger.info(f"Starting recursive loading from: {config.url}")
                
                parsed_url = urlparse(str(config.url))
                base_domain = parsed_url.netloc
                
                loader = RecursiveUrlLoader(
                    url=str(config.url),
                    max_depth=config.max_depth or 2,
                    extractor=langchain_docs_extractor,
                    prevent_outside=config.allowed_domains is None,
                    allowed_domains=config.allowed_domains or [base_domain],
                    use_async=True,
                    timeout=10,  # Increase timeout
                    check_response=lambda r: r.status_code == 200,
                    exclude_dirs=config.exclude_patterns or [
                        "*/login*", "*/signup*", "*/logout*", "*/admin*", "*/wp-admin*"
                    ],
                    headers={
                        'User-Agent': 'Mozilla/5.0 (compatible; DocumentIngestionBot/1.0)'
                    },
                    delay=1,  # 1 second delay between requests
                    retry_on_failure=True,
                    max_retries=2,
                )
                
                docs = loader.load()
                logger.info(f"Successfully loaded {len(docs)} documents via recursive crawling")
                
                if not docs:
                    logger.warning("No documents loaded via recursive crawling")
                    raise ValueError("No documents could be loaded from the website")
                    
                return docs
                
            except Exception as e:
                logger.error(f"Error in recursive loading: {e}")
                
                # Try a simpler approach if the complex loader fails
                try:
                    logger.info("Attempting simplified recursive loading...")
                    
                    # Create a simple web scraper as fallback
                    from langchain_community.document_loaders import AsyncHtmlLoader
                    from langchain_community.document_transformers import BeautifulSoupTransformer
                    
                    # Get initial page
                    loader = AsyncHtmlLoader([str(config.url)])
                    docs = loader.load()
                    
                    if docs:
                        # Extract main content
                        transformer = BeautifulSoupTransformer()
                        docs_transformed = transformer.transform_documents(
                            docs,
                            tags_to_extract=["article", "main", "div.content", "section"],
                            unwanted_tags=["header", "footer", "nav", "aside", "script", "style"],
                        )
                        
                        if docs_transformed:
                            logger.info(f"Loaded {len(docs_transformed)} documents via simplified approach")
                            return docs_transformed
                    
                    raise ValueError("Could not load any content from the website")
                    
                except Exception as fallback_error:
                    logger.error(f"Fallback loading also failed: {fallback_error}")
                    raise ValueError(f"Failed to load documents: {str(fallback_error)}")
    
        raise ValueError(f"Unsupported strategy: {config.strategy}")

    
    # def ingest_website(self, config: WebsiteConfig, force_update: bool = False) -> Dict[str, Any]:
    #     """Main ingestion method"""
    #     try:
    #         # Determine collection name
    #         collection_name = self._create_index_name(config)
            
    #         # Initialize vector store
    #         # vectorstore = Chroma(
    #         #     client=self.client,
    #         #     collection_name=collection_name,
    #         #     embedding_function=self.embedding,
    #         # )

    #         #weaviate initialize
            
    #         vectorstore = VectorStoreFactory.create_vector_store(
    #             store_type=VectorStoreType.WEAVIATE,
    #             embedding_model=self.embedding,
    #             **config
    #             )
            
    #             # Initialize record manager
    #         record_manager = SQLRecordManager(
    #                 f"chroma/{collection_name}",
    #                 db_url=os.getenv("RECORD_MANAGER_DB_URL")
    #             )
    #         record_manager.create_schema()
            
    #             # Load documents
    #         logger.info(f"Loading documents from {config.url} using {config.strategy} strategy")
    #         docs = self.load_documents(config)
    #         logger.info(f"Loaded {len(docs)} documents")
                
    #             # Split documents
    #         docs_transformed = self.text_splitter.split_documents(docs)
    #         docs_transformed = [
    #                 doc for doc in docs_transformed 
    #                 if len(doc.page_content) > 10
    #             ]
                
    #             # Ensure required metadata
    #         for doc in docs_transformed:
    #             if "source" not in doc.metadata:
    #                 doc.metadata["source"] = str(config.url)
    #             if "title" not in doc.metadata:
    #                 doc.metadata["title"] = config.name
                
    #             # Index documents
    #         indexing_stats = index(
    #                 docs_transformed,
    #                 record_manager,
    #                 vectorstore,
    #                 cleanup="full",
    #                 source_id_key="source",
    #                 force_update=force_update,
    #             )
                
    #             # Get collection statistics
    #         collection = self.client.get_collection(collection_name)
    #         num_vectors = collection.count()
                
    #         return {
    #                 "success": True,
    #                 "collection_name": collection_name,
    #                 "indexing_stats": indexing_stats,
    #                 "num_documents": len(docs),
    #                 "num_chunks": len(docs_transformed),
    #                 "num_vectors": num_vectors,
    #                 "strategy_used": config.strategy,
    #             }
            
    #     except Exception as e:
    #         logger.error(f"Error ingesting website {config.url}: {e}")
    #         return {
    #             "success": False,
    #             "error": str(e),
    #             "collection_name": getattr(config, 'index_name', 'unknown'),
    #         }
    def ingest_website(self, config: WebsiteConfig, force_update: bool = False) -> Dict[str, Any]:
        """Main ingestion method"""
        try:
            # Determine collection name
            collection_name = self._create_index_name(config)
            
            # Prepare config for vector store factory
            vector_store_config = {
                'index_name': collection_name,
                'collection_name': collection_name,  # For compatibility
            }
            
            # Add Weaviate credentials from config or environment
            
            # vector_store_config['weaviate_cluster_url'] = os.getenv('WEAVIATE_URL')
            # vector_store_config['weaviate_api_key'] = os.getenv('WEAVIATE_API_KEY')
            
            # Add user_id if available
            if hasattr(config, 'user_id'):
                vector_store_config['user_id'] = config.user_id
            
            # Validate required config
            # if not vector_store_config.get('weaviate_cluster_url'):
            #     raise ValueError("Weaviate cluster URL is required")
            # if not vector_store_config.get('weaviate_api_key'):
            #     raise ValueError("Weaviate API key is required")
            
            # Create vector store using factory
            with weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=weaviate.classes.init.Auth.api_key(WEAVIATE_API_KEY),
            skip_init_checks=True,
            ) as client:
                vectorstore = WeaviateVectorStore(
                    client=client,
                    embedding=self.embedding,
                    text_key="text",
                    index_name=config.index_name,
                    attributes=["source", "title", "user_id"]
                )
                # Initialize record manager
                record_namespace = f"weaviate/{collection_name}"
                record_manager = SQLRecordManager(
                    record_namespace,
                    db_url=os.getenv("RECORD_MANAGER_DB_URL", "sqlite:///record_manager_cache.db")
                )
                record_manager.create_schema()
                
                # Load documents
                logger.info(f"Loading documents from {config.url} using {config.strategy} strategy")
                docs = self.load_documents(config)
                logger.info(f"Loaded {len(docs)} documents")
                
                # Split documents
                docs_transformed = self.text_splitter.split_documents(docs)
                
                docs_transformed = [
                    doc for doc in docs_transformed
                    if len(doc.page_content) > 10
                ]
                allowed_attrs = ["source", "title", "user_id"]
                for doc in docs_transformed:
                    doc.metadata = {k: v for k, v in doc.metadata.items() if k in allowed_attrs}
            
                # Ensure required metadata
                for doc in docs_transformed:
                    if "source" not in doc.metadata:
                        doc.metadata["source"] = ""
                    if "title" not in doc.metadata:
                        doc.metadata["title"] = ""
                    # Add user_id if available
                    if hasattr(config, 'user_id') and config.user_id:
                        doc.metadata["user_id"] = config.user_id

                
                # Index documents
                indexing_stats = index(
                    docs_transformed,
                    record_manager,
                    vectorstore,
                    cleanup="full",
                    source_id_key="source",
                    force_update=True,
                    # key_encoder="sha256",
                )
                logger.info(f"Indexing stats: {indexing_stats}")
                # Get statistics from Weaviate
                # num_vectors = 0
                # if hasattr(vectorstore, 'client'):
                #     # Access the Weaviate client to get count
                #     try:
                #         # This depends on your WeaviateVectorStore implementation
                #         # You might need to add a method to get count
                #         num_vectors = vectorstore.get_document_count()
                #     except:
                #         # Fallback or estimate
                #         num_vectors = len(docs_transformed)
                num_vecs = (
                    client.collections.get(
                        config.index_name
                    )
                    .aggregate.over_all()
                    .total_count
                )
                
                logger.info(
                    f"General Guides and Tutorials now has this many vectors: {num_vecs}",)
                return {
                    "success": True,
                    "collection_name": collection_name,
                    "indexing_stats": indexing_stats,
                    "num_documents": len(docs),
                    "num_chunks": len(docs_transformed),
                    "num_vectors": num_vecs,
                    "strategy_used": config.strategy,
                    "vector_store": "weaviate",
        }
        
        except Exception as e:
            logger.error(f"Error ingesting website {config.url}: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "collection_name": collection_name if 'collection_name' in locals() else getattr(config, 'index_name', 'unknown'),
            }
        

# Singleton instance
ingestion_service = IngestionService()