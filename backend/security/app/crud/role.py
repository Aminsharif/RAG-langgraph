from sqlalchemy.orm import Session
from backend.security.app.models.user import Role, Permission
from backend.security.app.schemas.user import RoleCreate, RoleUpdate, PermissionCreate, PermissionUpdate
from typing import List
import uuid
def create_role(db: Session, role: RoleCreate) -> Role:
    db_role = Role(name=role.name)
    db.add(db_role)
    db.commit()
    db.refresh(db_role)
    return db_role

def get_role(db: Session, role_id: int) -> Role:
    return db.query(Role).filter(Role.id == uuid.UUID(role_id)).first()

def get_role_by_name(db: Session, name: str) -> Role:
    return db.query(Role).filter(Role.name == name).first()

def update_role(db: Session, role_id: str, role_update: RoleUpdate) -> Role:
    db_role = get_role(db, role_id)
    if not db_role:
        return None
    update_data = role_update.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_role, key, value)
    db.commit()
    db.refresh(db_role)
    return db_role

def delete_role(db: Session, role_id: str) -> bool:
    db_role = get_role(db, role_id)
    if not db_role:
        return False
    db.delete(db_role)
    db.commit()
    return True

def get_all_roles(db: Session, skip: int, limit: int) -> List[Role]:
    # result = db.query(Role).offset(skip).limit(limit).all()
    result = db.query(Role).all()
    return result

def create_permission(db: Session, permission: PermissionCreate) -> Permission:
    db_permission = Permission(name=permission.name)
    db.add(db_permission)
    db.commit()
    db.refresh(db_permission)
    return db_permission

def get_permission(db: Session, permission_id: str) -> Permission:
    return db.query(Permission).filter(Permission.id == uuid.UUID(permission_id)).first()

def update_permission(db: Session, permission_id: str, permission_update: PermissionUpdate) -> Permission:
    db_permission = get_permission(db, permission_id)
    if not db_permission:
        return None
    update_data = permission_update.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_permission, key, value)
    db.commit()
    db.refresh(db_permission)
    return db_permission

def delete_permission(db: Session, permission_id: str) -> bool:
    db_permission = get_permission(db, permission_id)
    if not db_permission:
        return False
    db.delete(db_permission)
    db.commit()
    return True

def get_all_permissions(db: Session, skip: int, limit: int) -> List[Permission]:
    # return db.query(Permission).offset(skip).limit(limit).all()
    return db.query(Permission).all()

def add_permission_to_role(db: Session, role: Role, permission: Permission) -> Role:
    if permission not in role.permissions:
        role.permissions.append(permission)
        db.commit()
        db.refresh(role)
    return role

def remove_permission_from_role(db: Session, role: Role, permission: Permission) -> Role:
    role.permissions.remove(permission)
    db.commit()
    db.refresh(role)
    return role