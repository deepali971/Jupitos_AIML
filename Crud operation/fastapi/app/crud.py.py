from sqlalchemy.orm import Session
from models import User
from datetime import date

def get_user(db: Session, user_id: int):
    return db.query(User).filter(User.id == user_id).first()

def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(User).offset(skip).limit(limit).all()

def create_user(db: Session, user: dict):
    dob = user["dob"]
    age = User.calculate_age(dob)
    db_user = User(username=user["username"], email=user["email"], dob=dob, age=age)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def update_user(db: Session, user_id: int, user: dict):
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user:
        update_data = user.dict(exclude_unset=True)
        if "username" in update_data:
            db_user.username = update_data["username"]
        if "email" in update_data:
            db_user.email = update_data["email"]
        if "dob" in update_data:
            db_user.dob = update_data["dob"]
            db_user.age = User.calculate_age(update_data["dob"])
        db.commit()
        db.refresh(db_user)
    return db_user

def delete_user(db: Session, user_id: int):
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user:
        db.delete(db_user)
        db.commit()
    return db_user

def search_users(db: Session, field: str, value: str):
    query = db.query(User)
    
    if field == "username":
        return query.filter(User.username.ilike(f"%{value}%")).all()
    elif field == "email":
        return query.filter(User.email.ilike(f"%{value}%")).all()
    elif field == "dob":
        return query.filter(User.dob == value).all()
    elif field == "age":
        return query.filter(User.age == int(value)).all()
    else:
        return []