from fastapi import FastAPI, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from models import User
from schemas import UserCreate, UserUpdate, User as UserSchema
from database import SessionLocal, engine
from crud import (
    get_user, get_users, create_user, 
    update_user, delete_user, search_users
)
from datetime import date
import models

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/users/", response_model=UserSchema)
def create_user_endpoint(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(
        (User.username == user.username) | 
        (User.email == user.email)
    ).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username or email already registered")
    return create_user(db=db, user=user.dict())

@app.get("/users/", response_model=list[UserSchema])
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
def update_user_endpoint(
    user_id: int, 
    user: UserUpdate, 
    db: Session = Depends(get_db)
):
    db_user = get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return update_user(db=db, user_id=user_id, user=user)

@app.delete("/users/{user_id}", response_model=UserSchema)
def delete_user_endpoint(user_id: int, db: Session = Depends(get_db)):
    db_user = delete_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@app.get("/users/search/", response_model=list[UserSchema])
def search_users_endpoint(
    field: str = Query(..., description="Field to search (username, email, dob, age)"),
    value: str = Query(..., description="Value to search for"),
    db: Session = Depends(get_db)
):
    return search_users(db, field, value)