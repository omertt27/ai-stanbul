from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database import SessionLocal
from models import Place

router = APIRouter(prefix="/places", tags=["Places"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/")
def get_places(db: Session = Depends(get_db)):
    return db.query(Place).all()
