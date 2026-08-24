"""User Authentication and Session Management for AI Resume Analyzer.

Provides secure password hashing, login throttling, and user profile management.
"""

import os
import json
import hashlib
import secrets
import time
from typing import Dict, Any, Tuple, Optional
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
USERS_FILE = os.path.join(DATA_DIR, "users.json")
os.makedirs(DATA_DIR, exist_ok=True)

MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_SECONDS = 15 * 60
PASSWORD_ITERATIONS = 200_000


class AuthManager:
    """Manages user registration, credential verification, throttling, and user storage."""

    @staticmethod
    def _hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """Hashes a password with a unique random salt using PBKDF2-HMAC-SHA256."""
        if salt is None:
            salt = secrets.token_hex(16)
        key = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            PASSWORD_ITERATIONS,
        )
        return key.hex(), salt

    @staticmethod
    def _legacy_hash_password(password: str, salt: str) -> str:
        """Compatibility helper for older SHA-256 + salt hashes."""
        return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_user_record(user_record: Dict[str, Any]) -> Dict[str, Any]:
        """Ensures records include safe defaults for lockout and scheme metadata."""
        user_record.setdefault("login_attempts", 0)
        user_record.setdefault("lockout_until", 0)
        user_record.setdefault("password_scheme", "pbkdf2_sha256")
        return user_record

    @classmethod
    def _load_users(cls) -> Dict[str, Any]:
        """Loads users dictionary from disk and strips any demo/test accounts."""
        if not os.path.exists(USERS_FILE):
            cls._save_users({})
            return {}

        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                users = json.load(f)
                if not isinstance(users, dict):
                    return {}

            for username in list(users.keys()):
                if username == "demo_user":
                    del users[username]
                    continue
                if isinstance(users[username], dict):
                    users[username] = cls._normalize_user_record(users[username])

            cls._save_users(users)
            return users
        except Exception:
            return {}

    @classmethod
    def _save_users(cls, users: Dict[str, Any]) -> None:
        """Saves users dictionary to JSON file with restrictive permissions."""
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2)
        try:
            os.chmod(USERS_FILE, 0o600)
        except OSError:
            pass

    @classmethod
    def register_user(
        cls,
        username: str,
        email: str,
        password: str,
        full_name: str = "",
        target_role: str = ""
    ) -> Tuple[bool, str]:
        """Registers a new user with validation and hashed credentials."""
        username = username.strip().lower()
        email = email.strip().lower()
        full_name = full_name.strip()

        if not username or len(username) < 3:
            return False, "Username must be at least 3 characters long."

        if not email or "@" not in email or "." not in email:
            return False, "Please enter a valid email address."

        if not password or len(password) < 8:
            return False, "Password must be at least 8 characters long."

        users = cls._load_users()

        if username in users:
            return False, "Username already exists. Please choose a different username."

        for user_data in users.values():
            if user_data.get("email") == email:
                return False, "An account with this email already exists."

        pwd_hash, salt = cls._hash_password(password)

        users[username] = {
            "username": username,
            "email": email,
            "full_name": full_name or username.capitalize(),
            "password_hash": pwd_hash,
            "salt": salt,
            "password_scheme": "pbkdf2_sha256",
            "created_at": datetime.now().isoformat(),
            "scans_count": 0,
            "target_role": target_role or "Software Engineer",
            "login_attempts": 0,
            "lockout_until": 0,
        }

        cls._save_users(users)
        return True, f"Account created successfully for {username}! You can now log in."

    @classmethod
    def authenticate_user(
        cls,
        username_or_email: str,
        password: str
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Authenticates a user by username or email and password."""
        identifier = username_or_email.strip().lower()
        if not identifier or not password:
            return False, "Please provide both username/email and password.", None

        users = cls._load_users()
        user_record = None

        if identifier in users:
            user_record = users[identifier]
        else:
            for u in users.values():
                if u.get("email") == identifier:
                    user_record = u
                    break

        if not user_record:
            return False, "Invalid username/email or password.", None

        user_record = cls._normalize_user_record(user_record)
        lockout_until = float(user_record.get("lockout_until", 0) or 0)
        if lockout_until and time.time() < lockout_until:
            remaining_seconds = max(1, int(lockout_until - time.time()))
            minutes, secs = divmod(remaining_seconds, 60)
            return False, f"Account temporarily locked. Try again in {minutes}m {secs}s.", None
        if lockout_until and time.time() >= lockout_until:
            user_record["lockout_until"] = 0
            user_record["login_attempts"] = 0

        stored_hash = user_record.get("password_hash")
        stored_salt = user_record.get("salt")
        if not stored_hash or not stored_salt:
            return False, "Invalid username/email or password.", None

        calc_hash, _ = cls._hash_password(password, stored_salt)
        valid_password = calc_hash == stored_hash

        if not valid_password:
            legacy_hash = cls._legacy_hash_password(password, stored_salt)
            if legacy_hash == stored_hash:
                valid_password = True
                migrated_hash, migrated_salt = cls._hash_password(password)
                user_record["password_hash"] = migrated_hash
                user_record["salt"] = migrated_salt
                user_record["password_scheme"] = "pbkdf2_sha256"

        if not valid_password:
            attempts = int(user_record.get("login_attempts", 0)) + 1
            user_record["login_attempts"] = attempts
            if attempts >= MAX_LOGIN_ATTEMPTS:
                user_record["lockout_until"] = time.time() + LOCKOUT_SECONDS
                user_record["login_attempts"] = MAX_LOGIN_ATTEMPTS
                cls._save_users(users)
                return False, "Too many failed attempts. Account locked for 15 minutes.", None
            remaining = MAX_LOGIN_ATTEMPTS - attempts
            cls._save_users(users)
            return False, f"Invalid username/email or password. {remaining} attempts remaining.", None

        user_record["login_attempts"] = 0
        user_record["lockout_until"] = 0
        username_key = user_record.get("username") or identifier
        users[username_key] = user_record
        cls._save_users(users)

        profile = {
            "username": user_record["username"],
            "email": user_record["email"],
            "full_name": user_record.get("full_name", user_record["username"]),
            "created_at": user_record.get("created_at", ""),
            "scans_count": user_record.get("scans_count", 0),
            "target_role": user_record.get("target_role", "Software Engineer")
        }
        return True, "Login successful!", profile

    @classmethod
    def increment_user_scans(cls, username: str) -> int:
        """Increments and saves the scan counter for a user."""
        users = cls._load_users()
        if username in users:
            users[username]["scans_count"] = users[username].get("scans_count", 0) + 1
            cls._save_users(users)
            return users[username]["scans_count"]
        return 1

