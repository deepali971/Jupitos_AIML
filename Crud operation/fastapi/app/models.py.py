from sqlalchemy import Column, Integer, String, Date
from sqlalchemy.sql import func
from datetime import date
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    dob = Column(Date)
    age = Column(Integer)

    @staticmethod
    def calculate_age(dob):
        today = date.today()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))