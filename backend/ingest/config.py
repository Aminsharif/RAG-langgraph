# config.py
from pydantic import BaseModel, HttpUrl
from typing import Optional, List
from enum import Enum

class IndexingStrategy(str, Enum):
    SITEMAP = "sitemap"
    RECURSIVE = "recursive"
    CUSTOM = "custom"

class WebsiteConfig(BaseModel):
    url: HttpUrl
    name: str
    index_name: str
    user_id: str
    strategy: IndexingStrategy = IndexingStrategy.SITEMAP
    filter_urls: Optional[List[str]] = None
    allowed_domains: Optional[List[str]] = None
    max_depth: Optional[int] = 2
    chunk_size: int = 4000
    chunk_overlap: int = 200