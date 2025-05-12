from passlib.context import CryptContext
from typing import Optional

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def authenticate_user(username: str, password: str) -> Optional[dict]:
    # In a real application, you would fetch the user from a database
    # This is just a simple example
    if username != "test_user":
        return None
    hashed_password = get_password_hash("test_password")
    if not verify_password(password, hashed_password):
        return None
    return {"username": username}