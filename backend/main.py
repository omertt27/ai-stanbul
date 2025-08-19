import fastapi
fastapi.__version__
from fastapi import FastAPI
app = FastAPI()
@app.get("/")
def read_root():
    return {"message": "Hello Istanbul AI Guide!"}
