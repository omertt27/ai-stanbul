from sqlalchemy import Column, Integer, String
from database import Base

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database import SessionLocal
from models import Museum

router = APIRouter(prefix="/museums", tags=["Museums"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/")
def get_museums(db: Session = Depends(get_db)):
    return db.query(Museum).all()

