from fastapi import APIRouter
from backend.security.app.api.v1.endpoints import auth
from backend.security.app.api.v1.endpoints.user_management import users, roles
from backend.security.app.api.v1.endpoints.vectordb_management import vector_db

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(users.router, prefix="/users", tags=["User Management"])
api_router.include_router(roles.router, prefix="/roles_permissions", tags=["Roles and Permissions"])
api_router.include_router(vector_db.router, prefix="/vector_db", tags=["Vector db Management"])