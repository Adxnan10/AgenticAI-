from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database.db_connection import Base, engine

# Routers:
from API.auth import router as auth_router
from API.chat import router as chat_router



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router, tags=["Authentication"])
app.include_router(chat_router, tags=["Chat"])

# Initialize tables
Base.metadata.create_all(bind=engine)