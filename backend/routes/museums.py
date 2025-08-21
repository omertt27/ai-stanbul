from sqlalchemy import Column, Integer, String
from database import Base

class Museum(Base):
    __tablename__ = "museums"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    location = Column(String, nullable=False)
