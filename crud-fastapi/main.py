from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from datetime import date

from models import User
from schemas import UserCreate, UserUpdate, User as UserSchema
from database import SessionLocal, engine
from crud import (
    get_user, get_user_by_email, get_user_by_username, 
    get_users, create_user, update_user, delete_user
)

# Create tables
User.metadata.create_all(bind=engine)

app = FastAPI()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/users/", response_model=UserSchema)
def create_new_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    db_user = get_user_by_username(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already taken")
    return create_user(db=db, user=user)

@app.get("/users/", response_model=List[UserSchema])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = get_users(db, skip=skip, limit=limit)
    return users

@app.get("/users/{user_id}", response_model=UserSchema)
def read_user(user_id: int, db: Session = Depends(get_db)):
    db_user = get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@app.put("/users/{user_id}", response_model=UserSchema)
def update_existing_user(user_id: int, user: UserUpdate, db: Session = Depends(get_db)):
    db_user = get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check if new email is already taken
    if user.email and user.email != db_user.email:
        existing_user = get_user_by_email(db, email=user.email)
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")
    
    # Check if new username is already taken
    if user.username and user.username != db_user.username:
        existing_user = get_user_by_username(db, username=user.username)
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already taken")
    
    return update_user(db=db, user_id=user_id, user_update=user)

@app.patch("/users/{user_id}", response_model=UserSchema)
def patch_existing_user(user_id: int, user: UserUpdate, db: Session = Depends(get_db)):
    db_user = get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check if new email is already taken
    if user.email and user.email != db_user.email:
        existing_user = get_user_by_email(db, email=user.email)
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")
    
    # Check if new username is already taken
    if user.username and user.username != db_user.username:
        existing_user = get_user_by_username(db, username=user.username)
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already taken")
    
    return update_user(db=db, user_id=user_id, user_update=user)

@app.delete("/users/{user_id}", response_model=UserSchema)
def delete_existing_user(user_id: int, db: Session = Depends(get_db)):
    db_user = get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return delete_user(db=db, user_id=user_id)
