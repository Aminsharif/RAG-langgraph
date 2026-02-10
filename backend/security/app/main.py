from fastapi import FastAPI
from backend.security.app.api.v1.api import api_router

app = FastAPI(title="Fast API - Roles and Permissions API")

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def root():
    return {"message": "Welcome to Fast API - Roles and Permissions API"}