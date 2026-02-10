# from pydantic import BaseModel, Field, EmailStr, validator
# from typing import Optional, List
# from datetime import datetime
# from app.models.user import Permission

# class UserBase(BaseModel):
#     username: str
#     email: Optional[EmailStr] = None
#     full_name: Optional[str] = None
#     roles: Optional[List[str]] = None

# class UserCreate(UserBase):
#     password: str  # Plain text password

# class UserUpdate(UserBase):
#     password: Optional[str] = None

# class RoleResponse(BaseModel):
#     id: int
#     name: str

# class UserResponse(UserBase):
#     id: int
#     created_at: datetime
#     last_login: Optional[datetime] = None
#     roles: List[RoleResponse] = []

#     class Config:
#         from_attributes = True

# class User(UserBase):
#     id: int
#     hashed_password: str
#     created_at: datetime
#     last_login: Optional[datetime] = None
#     roles: List[str] = []

#     class Config:
#         from_attributes = True

# class Token(BaseModel):
#     access_token: str
#     token_type: str

# class TokenData(BaseModel):
#     username: Optional[str] = None

# class RoleBase(BaseModel):
#     name: str

# class RoleCreate(RoleBase):
#     pass

# class RoleUpdate(RoleBase):
#     pass

# class Role(RoleBase):
#     id: int
#     permissions: List[str]

#     class Config:
#         from_attributes = True

#     @validator('permissions', pre=True, each_item=True)
#     def convert_permissions_to_strings(cls, perm):
#         if isinstance(perm, Permission):
#             return perm.name
#         return perm

# class PermissionBase(BaseModel):
#     name: str

# class PermissionCreate(PermissionBase):
#     pass

# class PermissionUpdate(PermissionBase):
#     pass

# class PermissionResponse(PermissionBase):
#     id: int

#     class Config:
#         from_attributes = True

from pydantic import BaseModel, Field, EmailStr, validator, UUID4
from typing import Optional, List
from datetime import datetime
from fastapi import Form

def get_login_form(
    email: str = Form(...),  
    password: str = Form(...),
    remember:bool = Form(False),
):
    return {"email": email, "password": password, "remember": remember}

# Base user schemas
class UserBase(BaseModel):
    username: str
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    roles: Optional[List[str]] = None

class UserCreate(UserBase):
    password: str

class UserUpdate(UserBase):
    password: Optional[str] = None

# Response schemas with UUID4
class RoleResponse(BaseModel):
    id: UUID4  
    name: str
    
    class Config:
        from_attributes = True

class PermissionResponse(BaseModel):
    id: UUID4  
    name: str
    description: Optional[str] = None
    
    class Config:
        from_attributes = True

class UserResponse(UserBase):
    id: UUID4  
    created_at: datetime
    last_login: Optional[datetime] = None
    roles: List[RoleResponse] = []
    
    class Config:
        from_attributes = True

class UserInDB(BaseModel):
    id: UUID4  
    username: str
    email: EmailStr
    full_name: str
    hashed_password: str
    created_at: datetime
    last_login: Optional[datetime] = None
    
    class Config:
        from_attributes = True

# Token schema
class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: UserResponse
    expires_at: int
    
class LoginRequest(BaseModel):
    email: EmailStr
    password: str
    remember: bool = False

class TokenData(BaseModel):
    username: Optional[str] = None

# Role schemas
class RoleBase(BaseModel):
    id: UUID4
    name: str

class RoleCreate(RoleBase):
    pass

class RoleUpdate(RoleBase):
    pass

class Role(RoleBase):
    id: UUID4  
    permissions: List[str] = []
    
    class Config:
        from_attributes = True

# Permission schemas
class PermissionBase(BaseModel):
    name: str

class PermissionCreate(PermissionBase):
    pass

class PermissionUpdate(PermissionBase):
    pass

class Permission(PermissionBase):
    id: UUID4  
    
    class Config:
        from_attributes = True

# For request parameters (if needed as strings in URLs)
class UserIDParam(BaseModel):
    user_id: UUID4  # FastAPI will convert string to UUID

class RefreshRequest(BaseModel):
    refresh_token: str

# Models for logout requests
class LogoutRequest(BaseModel):
    refresh_token: Optional[str] = None  # Optional for logout from specific session

class LogoutAllRequest(BaseModel):
    confirm: bool = True
# For relationships
class UserWithRoles(BaseModel):
    id: UUID4  
    username: str
    email: EmailStr
    roles: List[RoleResponse]
    
    class Config:
        from_attributes = True

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    expires_at: int
    user: UserResponse

class PermissionBase(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    
    class Config:
        from_attributes = True
