from sqlalchemy import Column, Integer, String, MetaData, Table
from databases import Database
from pydantic import BaseModel
from typing import Optional

DATABASE_URL = "postgresql+asyncpg://admin:admin@localhost:5432/analyzer-db"

database = Database(DATABASE_URL)
metadata = MetaData()

User = Table(
  "user",
  metadata,
  Column("id", Integer, primary_key=True),
  Column("name", String(50)),
  Column("email", String(100)),
)

# Pydantic models for API
class UserCreate(BaseModel):
    name: str
    email: str

class UserResponse(BaseModel):
    id: int
    name: str
    email: str

class UserUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None

