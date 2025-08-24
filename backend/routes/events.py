import requests

url = "https://api.apify.com/v2/datasets/uVMU3BQr2wa6Vrbm5/items?token=apify_api_rjVAb7BKZgz7EiP8cZdPf7mqBhFdSD3uNdRy"

# Send GET request
response = requests.get(url)

# Check for success
if response.status_code == 200:
    data = response.json()
    # Print each item
    for item in data:
        print(item)
else:
    print(f"Failed to fetch dataset: {response.status_code} - {response.text}")

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..database import SessionLocal
from ..models import Event

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


