from pydantic import BaseModel, EmailStr
from datetime import date
from typing import Optional

class UserBase(BaseModel):
    username: str
    email: EmailStr
    dob: date

class UserCreate(UserBase):
    pass

class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    dob: Optional[date] = None

class User(UserBase):
    id: int
    age: int

    class Config:
        orm_mode = True