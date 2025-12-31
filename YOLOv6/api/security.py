import hashlib

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def verify_password(plain_password: str, stored_hash: str) -> bool:
    return hash_password(plain_password) == stored_hash
