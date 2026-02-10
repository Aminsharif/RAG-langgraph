import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    SECRET_KEY: str = os.getenv('SECRET_KEY', 'your_secret_key@$%QRT$%')
    REFRESH_SECRET_KEY: str = os.getenv('REFRESH_SECRET_KEY', 'your_secret_key#$^W$EDGR')
    REFRESH_TOKEN_EXPIRE_DAYS: int = os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "7")
    ALGORITHM: str = os.environ.get("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "30")
    DATABASE_URL: str = os.getenv('DATABASE_URL', 'postgresql://username:password@localhost:5432/test')
    FILE_UPLOAD_DIR: str = os.getenv('FILE_UPLOAD_DIR', './files')

    class Config:
        case_sensitive = True

settings = Settings()