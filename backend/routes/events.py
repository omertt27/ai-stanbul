from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database import SessionLocal
from models import Event

router = APIRouter(prefix="/events", tags=["Events"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/")
def get_events(db: Session = Depends(get_db)):
    return db.query(Event).all()


