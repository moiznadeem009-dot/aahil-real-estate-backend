from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr, ConfigDict
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    Index,
    text,
)
from sqlalchemy.orm import (
    declarative_base,
    sessionmaker,
    relationship,
)

from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
import re
import os
import base64
import hashlib
import hmac
import json
import secrets

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


if load_dotenv:
    load_dotenv()


# =========================================================
# CONFIGURATION
# =========================================================

APP_VERSION = "20.0.0"

APP_ENV = os.getenv("APP_ENV", "development").strip().lower()

AUTH_SECRET = os.getenv("AUTH_SECRET", "").strip()

if not AUTH_SECRET:
    if APP_ENV in {"production", "prod"}:
        raise RuntimeError(
            "AUTH_SECRET is required when APP_ENV=production."
        )

    AUTH_SECRET = "development-only-change-this-secret"
    print(
        "WARNING: AUTH_SECRET is using the development fallback. "
        "Set AUTH_SECRET in your .env before production."
    )

AUTH_TOKEN_EXPIRE_MINUTES = 60 * 24

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://127.0.0.1:11434/api/chat",
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:3b",
)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./chatbot.db",
)

MAX_COMPARISON_PROPERTIES = 8
MAX_HISTORY_MESSAGES = 10
MAX_MEMORY_ITEMS = 50


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="Enterprise Property AI Chatbot",
    version=APP_VERSION,
    description=(
        "Property AI chatbot with authentication, "
        "memory, conversations, property search, "
        "property comparison, lead capture and agent dashboard."
    ),
)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://127.0.0.1:3000,http://localhost:3000,http://127.0.0.1:5173,http://localhost:5173",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# DATABASE
# =========================================================

engine_kwargs = {
    "pool_pre_ping": True,
}

if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {
        "check_same_thread": False
    }

engine = create_engine(
    DATABASE_URL,
    **engine_kwargs,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


# =========================================================
# DATABASE MODELS
# =========================================================

class User(Base):

    __tablename__ = "users"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    name = Column(
        String(100),
        nullable=False,
    )

    email = Column(
        String(255),
        unique=True,
        nullable=True,
        index=True,
    )

    password_hash = Column(
        String(512),
        nullable=True,
    )

    role = Column(
        String(50),
        nullable=False,
        default="customer",
    )

    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    conversations = relationship(
        "Conversation",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    memories = relationship(
        "Memory",
        back_populates="user",
        cascade="all, delete-orphan",
    )


class Conversation(Base):

    __tablename__ = "conversations"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    user_id = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    user = relationship(
        "User",
        back_populates="conversations",
    )

    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
    )

    active_property = relationship(
        "ActiveProperty",
        back_populates="conversation",
        uselist=False,
        cascade="all, delete-orphan",
    )

    comparison = relationship(
        "Comparison",
        back_populates="conversation",
        uselist=False,
        cascade="all, delete-orphan",
    )


class Message(Base):

    __tablename__ = "messages"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    conversation_id = Column(
        Integer,
        ForeignKey("conversations.id"),
        nullable=False,
        index=True,
    )

    user_message = Column(
        Text,
        nullable=False,
    )

    bot_reply = Column(
        Text,
        nullable=False,
    )

    timestamp = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    conversation = relationship(
        "Conversation",
        back_populates="messages",
    )


class Memory(Base):

    __tablename__ = "memories"

    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "key",
            name="uq_user_memory_key",
        ),
        Index(
            "ix_memories_user_id_key",
            "user_id",
            "key",
        ),
    )

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    user_id = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    key = Column(
        String(100),
        nullable=False,
    )

    value = Column(
        Text,
        nullable=False,
    )

    timestamp = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    user = relationship(
        "User",
        back_populates="memories",
    )


class Property(Base):

    __tablename__ = "properties"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    title = Column(
        String(200),
        nullable=False,
    )

    property_type = Column(
        String(50),
        nullable=False,
        default="house",
    )

    purpose = Column(
        String(50),
        nullable=False,
        default="sale",
    )

    location = Column(
        String(200),
        nullable=False,
    )

    price = Column(
        Integer,
        nullable=False,
    )

    size_marla = Column(
        Integer,
        nullable=False,
    )

    bedrooms = Column(
        Integer,
        nullable=False,
    )

    bathrooms = Column(
        Integer,
        nullable=False,
    )

    description = Column(
        Text,
        nullable=False,
        default="",
    )

    status = Column(
        String(50),
        nullable=False,
        default="available",
    )

    contact_name = Column(
        String(100),
        nullable=False,
    )

    contact_phone = Column(
        String(50),
        nullable=False,
    )

    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class ActiveProperty(Base):

    __tablename__ = "active_properties"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    conversation_id = Column(
        Integer,
        ForeignKey("conversations.id"),
        unique=True,
        nullable=False,
    )

    property_id = Column(
        Integer,
        ForeignKey("properties.id"),
        nullable=False,
    )

    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    conversation = relationship(
        "Conversation",
        back_populates="active_property",
    )

    property = relationship(
        "Property",
    )


class Lead(Base):
    __tablename__ = "leads"

    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    conversation_id = Column(
        Integer,
        ForeignKey("conversations.id"),
        nullable=False,
        index=True,
    )

    name = Column(String(100), nullable=True)
    phone = Column(String(50), nullable=True)
    email = Column(String(255), nullable=True)
    budget = Column(Integer, nullable=True)
    preferred_location = Column(String(200), nullable=True)
    purpose = Column(String(50), nullable=True)
    property_type = Column(String(50), nullable=True)
    bedrooms = Column(Integer, nullable=True)
    notes = Column(Text, nullable=True)
    status = Column(String(50), nullable=False, default="new")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)


class Comparison(Base):

    __tablename__ = "comparisons"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    conversation_id = Column(
        Integer,
        ForeignKey("conversations.id"),
        unique=True,
        nullable=False,
    )

    property_ids = Column(
        Text,
        nullable=False,
    )

    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    conversation = relationship(
        "Conversation",
        back_populates="comparison",
    )


# =========================================================
# CREATE TABLES
# =========================================================

Base.metadata.create_all(bind=engine)


# =========================================================
# PYDANTIC MODELS
# =========================================================

class ChatRequest(BaseModel):

    message: str = Field(
        min_length=1,
        max_length=5000,
    )


class ChatResponse(BaseModel):

    reply: str


class RegisterRequest(BaseModel):

    name: str = Field(
        min_length=2,
        max_length=100,
    )

    email: EmailStr

    password: str = Field(
        min_length=8,
        max_length=128,
    )


class AgentRegisterRequest(BaseModel):

    name: str = Field(
        min_length=2,
        max_length=100,
    )

    email: EmailStr

    password: str = Field(
        min_length=8,
        max_length=128,
    )

    registration_key: str = Field(
        min_length=1,
        max_length=256,
    )


class LoginRequest(BaseModel):

    email: EmailStr

    password: str = Field(
        min_length=8,
        max_length=128,
    )


class MemoryCreate(BaseModel):

    key: str = Field(
        min_length=1,
        max_length=100,
    )

    value: str = Field(
        min_length=1,
        max_length=5000,
    )


class LeadCreate(BaseModel):
    name: Optional[str] = Field(default=None, max_length=100)
    phone: Optional[str] = Field(default=None, max_length=50)
    email: Optional[EmailStr] = None
    budget: Optional[int] = Field(default=None, gt=0)
    preferred_location: Optional[str] = Field(default=None, max_length=200)
    purpose: Optional[str] = Field(default=None, max_length=50)
    property_type: Optional[str] = Field(default=None, max_length=50)
    bedrooms: Optional[int] = Field(default=None, ge=0)
    notes: Optional[str] = Field(default=None, max_length=5000)
    status: str = Field(default="new", max_length=50)


class PropertyCreate(BaseModel):

    title: str = Field(
        min_length=1,
        max_length=200,
    )

    property_type: str = "house"

    purpose: str = "sale"

    location: str = Field(
        min_length=1,
        max_length=200,
    )

    price: int = Field(
        gt=0,
    )

    size_marla: int = Field(
        gt=0,
    )

    bedrooms: int = Field(
        ge=0,
    )

    bathrooms: int = Field(
        ge=0,
    )

    description: str = ""

    status: str = "available"

    contact_name: str = Field(
        min_length=1,
        max_length=100,
    )

    contact_phone: str = Field(
        min_length=1,
        max_length=50,
    )


class PropertyUpdate(PropertyCreate):
    pass


# =========================================================
# AUTHENTICATION
# =========================================================

security = HTTPBearer(
    auto_error=False
)


def clean_text(value):

    if value is None:
        return ""

    return str(value).strip()


def normalize_email(email):

    return clean_text(email).lower()


def hash_password(password):

    salt = secrets.token_bytes(16)

    iterations = 310000

    derived_key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
    )

    return (
        f"pbkdf2_sha256${iterations}$"
        f"{base64.urlsafe_b64encode(salt).decode()}$"
        f"{base64.urlsafe_b64encode(derived_key).decode()}"
    )


def verify_password(
    password,
    stored_hash,
):

    try:

        algorithm, iterations, salt_b64, hash_b64 = (
            stored_hash.split("$", 3)
        )

        if algorithm != "pbkdf2_sha256":
            return False

        salt = base64.urlsafe_b64decode(
            salt_b64.encode()
        )

        expected = base64.urlsafe_b64decode(
            hash_b64.encode()
        )

        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            int(iterations),
        )

        return hmac.compare_digest(
            actual,
            expected,
        )

    except Exception:

        return False


def _b64url(data):

    return (
        base64.urlsafe_b64encode(data)
        .rstrip(b"=")
        .decode("ascii")
    )


def _b64url_decode(value):

    padding = "=" * (-len(value) % 4)

    return base64.urlsafe_b64decode(
        (value + padding).encode("ascii")
    )


def create_access_token(
    user_id,
    role,
):

    now = datetime.now(timezone.utc)

    payload = {
        "sub": str(user_id),
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int(
            (
                now
                + timedelta(
                    minutes=AUTH_TOKEN_EXPIRE_MINUTES
                )
            ).timestamp()
        ),
    }

    header = {
        "alg": "HS256",
        "typ": "JWT",
    }

    encoded_header = _b64url(
        json.dumps(
            header,
            separators=(",", ":"),
        ).encode("utf-8")
    )

    encoded_payload = _b64url(
        json.dumps(
            payload,
            separators=(",", ":"),
        ).encode("utf-8")
    )

    signing_input = (
        f"{encoded_header}.{encoded_payload}"
        .encode("ascii")
    )

    signature = hmac.new(
        AUTH_SECRET.encode("utf-8"),
        signing_input,
        hashlib.sha256,
    ).digest()

    return (
        f"{encoded_header}."
        f"{encoded_payload}."
        f"{_b64url(signature)}"
    )


def decode_access_token(token):

    try:

        parts = token.split(".")

        if len(parts) != 3:
            return None

        (
            encoded_header,
            encoded_payload,
            encoded_signature,
        ) = parts

        signing_input = (
            f"{encoded_header}.{encoded_payload}"
            .encode("ascii")
        )

        expected_signature = hmac.new(
            AUTH_SECRET.encode("utf-8"),
            signing_input,
            hashlib.sha256,
        ).digest()

        actual_signature = _b64url_decode(
            encoded_signature
        )

        if not hmac.compare_digest(
            expected_signature,
            actual_signature,
        ):
            return None

        header = json.loads(
            _b64url_decode(
                encoded_header
            ).decode("utf-8")
        )

        if header.get("alg") != "HS256":
            return None

        payload = json.loads(
            _b64url_decode(
                encoded_payload
            ).decode("utf-8")
        )

        exp = int(
            payload.get("exp", 0)
        )

        if exp <= int(
            datetime.now(timezone.utc).timestamp()
        ):
            return None

        if not payload.get("sub"):
            return None

        return payload

    except Exception:

        return None


def get_current_user(
    credentials: Optional[
        HTTPAuthorizationCredentials
    ] = Depends(security),
):

    if (
        not credentials
        or credentials.scheme.lower()
        != "bearer"
    ):

        raise HTTPException(
            status_code=401,
            detail=(
                "Authentication required. "
                "Use Bearer token."
            ),
            headers={
                "WWW-Authenticate": "Bearer"
            },
        )

    payload = decode_access_token(
        credentials.credentials
    )

    if not payload:

        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token.",
            headers={
                "WWW-Authenticate": "Bearer"
            },
        )

    try:

        return int(
            payload["sub"]
        )

    except Exception:

        raise HTTPException(
            status_code=401,
            detail="Invalid token subject.",
            headers={
                "WWW-Authenticate": "Bearer"
            },
        )


def require_roles(*allowed_roles):

    def dependency(
        current_user_id: int = Depends(
            get_current_user
        ),
    ):

        db = SessionLocal()

        try:

            user = (
                db.query(User)
                .filter(
                    User.id
                    == current_user_id
                )
                .first()
            )

            if not user:

                raise HTTPException(
                    status_code=401,
                    detail=(
                        "User account "
                        "no longer exists."
                    ),
                )

            role = user.role or "customer"

            if role not in allowed_roles:

                raise HTTPException(
                    status_code=403,
                    detail=(
                        "You do not have permission "
                        "for this action."
                    ),
                )

            return current_user_id

        finally:

            db.close()

    return dependency


# =========================================================
# GENERAL HELPERS
# =========================================================

def money_lakh(price):

    try:

        return round(
            float(price) / 100000,
            2,
        )

    except Exception:

        return 0


def property_to_dict(property_obj):

    if not property_obj:
        return None

    return {
        "id": property_obj.id,
        "title": property_obj.title,
        "property_type": property_obj.property_type,
        "purpose": property_obj.purpose,
        "location": property_obj.location,
        "price": property_obj.price,
        "price_lakh": money_lakh(
            property_obj.price
        ),
        "size_marla": property_obj.size_marla,
        "bedrooms": property_obj.bedrooms,
        "bathrooms": property_obj.bathrooms,
        "description": property_obj.description,
        "status": property_obj.status,
        "contact_name": property_obj.contact_name,
        "contact_phone": property_obj.contact_phone,
        "created_at": property_obj.created_at,
    }


# =========================================================
# MEMORY
# =========================================================

BAD_MEMORY_VALUES = {
    "kya",
    "hai",
    "hain",
    "ka",
    "ki",
    "ke",
    "mera",
    "name",
    "naam",
    "what",
    "is",
    "my",
}


def save_memory(
    db,
    user_id,
    key,
    value,
):

    key = clean_text(key).lower()
    value = clean_text(value)

    if not key or not value:
        return None

    memory = (
        db.query(Memory)
        .filter(
            Memory.user_id == user_id,
            Memory.key == key,
        )
        .first()
    )

    if memory:

        memory.value = value
        memory.timestamp = datetime.now(timezone.utc)

    else:

        memory = Memory(
            user_id=user_id,
            key=key,
            value=value,
        )

        db.add(memory)

    db.commit()
    db.refresh(memory)

    return memory


def get_memory(
    db,
    user_id,
    key,
):

    key = clean_text(key).lower()

    return (
        db.query(Memory)
        .filter(
            Memory.user_id == user_id,
            Memory.key == key,
        )
        .first()
    )


def get_memory_value(
    db,
    user_id,
    key,
):

    memory = get_memory(
        db,
        user_id,
        key,
    )

    if not memory:
        return None

    value = clean_text(
        memory.value
    )

    if not value:
        return None

    if value.lower() in BAD_MEMORY_VALUES:
        return None

    return value


def extract_memories(
    db,
    user_id,
    message,
):

    text = clean_text(message)

    name_patterns = [
        r"\bmera\s+naam\s+([a-zA-Z][a-zA-Z'-]*)\s+hai\b",
        r"\bmera\s+name\s+([a-zA-Z][a-zA-Z'-]*)\s+hai\b",
        r"\bmy\s+name\s+is\s+([a-zA-Z][a-zA-Z'-]*)\b",
        r"\bname\s+is\s+([a-zA-Z][a-zA-Z'-]*)\b",
    ]

    for pattern in name_patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE,
        )

        if match:

            name = clean_text(
                match.group(1)
            )

            if (
                name.lower()
                not in BAD_MEMORY_VALUES
            ):

                save_memory(
                    db,
                    user_id,
                    "name",
                    name,
                )

                break

    learning_patterns = [
        r"\bmain\s+([a-zA-Z0-9+#.\-]+)\s+seekh\s+raha\s+hoon\b",
        r"\bmain\s+([a-zA-Z0-9+#.\-]+)\s+seekh\s+raha\s+hun\b",
        r"\bmain\s+([a-zA-Z0-9+#.\-]+)\s+seekh\s+raha\b",
        r"\bi\s+am\s+learning\s+([a-zA-Z0-9+#.\-]+)\b",
    ]

    for pattern in learning_patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE,
        )

        if match:

            learning = clean_text(
                match.group(1)
            ).rstrip(
                ".,!? "
            )

            if (
                learning.lower()
                not in BAD_MEMORY_VALUES
            ):

                save_memory(
                    db,
                    user_id,
                    "learning",
                    learning,
                )

                break


def is_name_question(text):

    lower = clean_text(
        text
    ).lower()

    patterns = [
        "mera naam kya hai",
        "mera name kya hai",
        "what is my name",
        "what's my name",
        "whats my name",
        "do you know my name",
        "what do you call me",
    ]

    return any(
        pattern in lower
        for pattern in patterns
    )


def is_learning_question(text):

    lower = clean_text(
        text
    ).lower()

    patterns = [
        "main kya seekh raha hoon",
        "main kya seekh raha hun",
        "main kya seekh raha",
        "main kya learn kar raha hoon",
        "main kya learning kar raha hoon",
        "what am i learning",
        "what do i learn",
        "what am i studying",
    ]

    return any(
        pattern in lower
        for pattern in patterns
    )


def answer_memory_question(
    db,
    user_id,
    message,
):

    if is_name_question(message):

        name = get_memory_value(
            db,
            user_id,
            "name",
        )

        if name:

            return (
                f"Tumhara naam {name} hai."
            )

        return (
            "Tumne abhi mujhe apna "
            "naam nahi bataya."
        )

    if is_learning_question(message):

        learning = get_memory_value(
            db,
            user_id,
            "learning",
        )

        if learning:

            return (
                f"Tum {learning} "
                "seekh rahe ho."
            )

        return (
            "Tum kya seekh rahe ho, "
            "ye mujhe abhi yaad nahi."
        )

    return None


# =========================================================
# PROPERTY STATE
# =========================================================

def get_active_property(
    db,
    conversation_id,
):

    active = (
        db.query(ActiveProperty)
        .filter(
            ActiveProperty.conversation_id
            == conversation_id
        )
        .first()
    )

    if not active:
        return None

    return (
        db.query(Property)
        .filter(
            Property.id
            == active.property_id
        )
        .first()
    )


def set_active_property(
    db,
    conversation_id,
    property_id,
):

    property_obj = (
        db.query(Property)
        .filter(
            Property.id
            == property_id
        )
        .first()
    )

    if not property_obj:
        return None

    active = (
        db.query(ActiveProperty)
        .filter(
            ActiveProperty.conversation_id
            == conversation_id
        )
        .first()
    )

    if active:

        active.property_id = property_id
        active.updated_at = datetime.now(timezone.utc)

    else:

        active = ActiveProperty(
            conversation_id=conversation_id,
            property_id=property_id,
        )

        db.add(active)

    db.commit()
    db.refresh(active)

    return property_obj


# =========================================================
# COMPARISON
# =========================================================

def set_comparison(
    db,
    conversation_id,
    properties,
):

    if not properties:
        return

    properties = list(properties)[:MAX_COMPARISON_PROPERTIES]

    ids = ",".join(
        str(property_obj.id)
        for property_obj in properties
    )

    comparison = (
        db.query(Comparison)
        .filter(
            Comparison.conversation_id
            == conversation_id
        )
        .first()
    )

    if comparison:

        comparison.property_ids = ids
        comparison.updated_at = datetime.now(timezone.utc)

    else:

        comparison = Comparison(
            conversation_id=conversation_id,
            property_ids=ids,
        )

        db.add(comparison)

    db.commit()


def get_comparison_properties(
    db,
    conversation_id,
):

    comparison = (
        db.query(Comparison)
        .filter(
            Comparison.conversation_id
            == conversation_id
        )
        .first()
    )

    if not comparison:
        return []

    ids = []

    for item in (
        comparison.property_ids or ""
    ).split(","):

        item = item.strip()

        if item.isdigit():

            property_id = int(item)

            if property_id not in ids:
                ids.append(property_id)

    if not ids:
        return []

    properties = (
        db.query(Property)
        .filter(
            Property.id.in_(ids)
        )
        .all()
    )

    property_map = {
        item.id: item
        for item in properties
    }

    return [
        property_map[property_id]
        for property_id in ids
        if property_id in property_map
    ]


# =========================================================
# PROPERTY ID EXTRACTION
# =========================================================

def extract_property_ids(message):

    lower = clean_text(
        message
    ).lower()

    patterns = [

        r"\bid\s*(\d+)\s*(?:aur|and|&|,)\s*(?:id\s*)?(\d+)\b",

        r"\bproperty\s*(\d+)\s*(?:aur|and|&|,)\s*(?:property\s*)?(\d+)\b",

        r"\bproperties\s*(\d+)\s*(?:aur|and|&|,)\s*(?:properties\s*)?(\d+)\b",

        r"\b(\d+)\s*(?:aur|and|&|,)\s*(\d+)\s*(?:compare|comparison|muqabla|mukabla)\b",

        r"\b(\d+)\s+(?:compare|comparison|muqabla|mukabla)\s+(\d+)\b",

        r"\b(?:compare|comparison|muqabla|mukabla)\s+(?:property\s*)?(\d+)\s*(?:aur|and|&|,|vs|versus)\s*(?:property\s*)?(\d+)\b",

        r"\b(?:property|properties)\s*(\d+)\s*(?:vs|versus)\s*(?:property|properties)?\s*(\d+)\b",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            lower,
            re.IGNORECASE,
        )

        if match:

            first_id = int(
                match.group(1)
            )

            second_id = int(
                match.group(2)
            )

            if first_id == second_id:
                return [first_id]

            return [
                first_id,
                second_id,
            ]

    return []


def get_properties_by_ids(
    db,
    property_ids,
):

    if not property_ids:
        return []

    properties = (
        db.query(Property)
        .filter(
            Property.id.in_(property_ids)
        )
        .all()
    )

    property_map = {
        item.id: item
        for item in properties
    }

    return [
        property_map[property_id]
        for property_id in property_ids
        if property_id in property_map
    ]


# =========================================================
# PRICE PARSING
# =========================================================

def parse_pakistani_amounts(text):
    """Return price amounts in PKR for common lakh/crore phrasing."""

    amount_pattern = re.compile(
        r"(\d+(?:\.\d+)?)\s*"
        r"(crore|crores|cr|lakh|lakhs|lac|lacs)\b",
        re.IGNORECASE,
    )

    values = []

    for match in amount_pattern.finditer(text):
        number = float(match.group(1))
        unit = match.group(2).lower()

        if unit in {"crore", "crores", "cr"}:
            value = number * 10_000_000
        else:
            value = number * 100_000

        values.append((match.start(), match.end(), int(value)))

    return values


# =========================================================
# PROPERTY SEARCH
# =========================================================

def search_properties(
    db,
    message,
):

    lower = clean_text(
        message
    ).lower()

    query = (
        db.query(Property)
        .filter(
            Property.status.ilike(
                "available"
            )
        )
    )

    # -----------------------------------------------------
    # LOCATION
    # -----------------------------------------------------

    known_locations = [
        "bahawalpur",
        "lahore",
        "multan",
        "islamabad",
        "rawalpindi",
        "karachi",
        "faisalabad",
        "sahiwal",
        "rahim yar khan",
        "lodhran",
        "vehari",
        "khanewal",
        "pakpattan",
        "dera ghazi khan",
        "dg khan",
    ]

    location = None

    for item in known_locations:

        if item in lower:

            location = item
            break

    if location:

        query = query.filter(
            Property.location.ilike(
                f"%{location}%"
            )
        )

    # -----------------------------------------------------
    # PROPERTY TYPE
    # -----------------------------------------------------

    if any(
        word in lower
        for word in [
            "house",
            "ghar",
            "home",
            "makan",
        ]
    ):

        query = query.filter(
            Property.property_type.ilike(
                "%house%"
            )
        )

    elif any(
        word in lower
        for word in [
            "plot",
            "zameen",
            "land",
        ]
    ):

        query = query.filter(
            Property.property_type.ilike(
                "%plot%"
            )
        )

    elif any(
        word in lower
        for word in [
            "flat",
            "apartment",
            "portion",
        ]
    ):

        query = query.filter(
            Property.property_type.ilike(
                "%flat%"
            )
        )

    # -----------------------------------------------------
    # PURPOSE
    # -----------------------------------------------------

    is_rent = any(
        word in lower
        for word in [
            "rent",
            "rental",
            "kiraye",
            "kiraya",
            "lease",
        ]
    )

    is_sale = any(
        word in lower
        for word in [
            "sale",
            "buy",
            "purchase",
            "khareed",
            "kharid",
            "kharidna",
        ]
    )

    if is_rent and not is_sale:

        query = query.filter(
            Property.purpose.ilike(
                "%rent%"
            )
        )

    elif is_sale and not is_rent:

        query = query.filter(
            Property.purpose.ilike(
                "%sale%"
            )
        )

    # -----------------------------------------------------
    # SIZE
    # -----------------------------------------------------

    size_match = re.search(
        r"\b(\d+(?:\.\d+)?)\s*marla\b",
        lower,
        re.IGNORECASE,
    )

    if size_match:

        size = float(
            size_match.group(1)
        )

        query = query.filter(
            Property.size_marla
            == int(size)
        )

    # -----------------------------------------------------
    # BEDROOMS
    # -----------------------------------------------------

    bedroom_match = re.search(
        r"\b(\d+)\s*(?:bedrooms?|beds?)\b",
        lower,
        re.IGNORECASE,
    )

    if bedroom_match:

        bedrooms = int(
            bedroom_match.group(1)
        )

        query = query.filter(
            Property.bedrooms
            == bedrooms
        )

    # -----------------------------------------------------
    # BATHROOMS
    # -----------------------------------------------------

    bathroom_match = re.search(
        r"\b(\d+)\s*(?:bathrooms?|baths?)\b",
        lower,
        re.IGNORECASE,
    )

    if bathroom_match:

        bathrooms = int(
            bathroom_match.group(1)
        )

        query = query.filter(
            Property.bathrooms
            == bathrooms
        )

    # -----------------------------------------------------
    # PRICE / BUDGET
    # -----------------------------------------------------

    amounts = parse_pakistani_amounts(lower)

    if len(amounts) >= 2 and re.search(
        r"(?:to|[-–—]|and|aur|between|se)"
        , lower,
        re.IGNORECASE,
    ):
        minimum_price = min(
            amounts[0][2],
            amounts[1][2],
        )
        maximum_price = max(
            amounts[0][2],
            amounts[1][2],
        )

        query = query.filter(
            Property.price >= minimum_price,
            Property.price <= maximum_price,
        )

    elif amounts:
        # A single stated amount is treated as a maximum budget.
        budget_price = amounts[0][2]
        query = query.filter(
            Property.price <= budget_price
        )

    return (
        query
        .order_by(
            Property.price.asc()
        )
        .all()
    )


# =========================================================
# PROPERTY FORMATTERS
# =========================================================

def format_property_list(
    properties,
):

    if not properties:

        return (
            "Mujhe tumhari requirements ke "
            "mutabiq koi available property nahi mili."
        )

    lines = [
        f"Mujhe {len(properties)} matching "
        "properties mili hain:"
    ]

    for index, property_obj in enumerate(
        properties,
        start=1,
    ):

        purpose_text = (
            "Rent"
            if "rent"
            in property_obj.purpose.lower()
            else "Sale"
        )

        lines.append(
            f"\n{index}. "
            f"{property_obj.title}"
        )

        lines.append(
            f"   ID: {property_obj.id}"
        )

        lines.append(
            f"   Purpose: {purpose_text}"
        )

        lines.append(
            f"   Price: "
            f"{money_lakh(property_obj.price)} lakh"
        )

        lines.append(
            f"   Size: "
            f"{property_obj.size_marla} marla"
        )

        lines.append(
            f"   Bedrooms: "
            f"{property_obj.bedrooms}"
        )

        lines.append(
            f"   Bathrooms: "
            f"{property_obj.bathrooms}"
        )

        lines.append(
            f"   Location: "
            f"{property_obj.location}"
        )

    return "\n".join(lines)


def format_property_details(
    property_obj,
):

    if not property_obj:
        return "Property information available nahi hai."

    return (
        f"{property_obj.title}\n\n"

        f"Property ID: "
        f"{property_obj.id}\n"

        f"Property type: "
        f"{property_obj.property_type}\n"

        f"Purpose: "
        f"{property_obj.purpose}\n"

        f"Location: "
        f"{property_obj.location}\n"

        f"Price: "
        f"{money_lakh(property_obj.price)} lakh\n"

        f"Size: "
        f"{property_obj.size_marla} marla\n"

        f"Bedrooms: "
        f"{property_obj.bedrooms}\n"

        f"Bathrooms: "
        f"{property_obj.bathrooms}\n"

        f"Description: "
        f"{property_obj.description}\n"

        f"Status: "
        f"{property_obj.status}\n"

        f"Owner: "
        f"{property_obj.contact_name}\n"

        f"Contact: "
        f"{property_obj.contact_phone}"
    )


# =========================================================
# LEAD CAPTURE
# =========================================================

def extract_phone(text):
    match = re.search(
        r"(?<!\d)(?:\+92|0092|92|0)?[-\s]?(?:3\d{2})[-\s]?\d{3}[-\s]?\d{4}(?!\d)",
        clean_text(text),
    )
    if not match:
        return None

    phone = re.sub(r"[^0-9+]", "", match.group(0))

    if phone.startswith("0092"):
        phone = "+" + phone[2:]
    elif phone.startswith("92") and not phone.startswith("+92"):
        phone = "+" + phone
    elif phone.startswith("03"):
        phone = "+92" + phone[1:]

    return phone


def extract_lead_name(text):
    patterns = [
        r"\bmera\s+naam\s+([a-zA-Z][a-zA-Z' -]{1,50})\s+hai\b",
        r"\bmera\s+name\s+([a-zA-Z][a-zA-Z' -]{1,50})\s+hai\b",
        r"\bmy\s+name\s+is\s+([a-zA-Z][a-zA-Z' -]{1,50})\b",
        r"\bmain\s+([a-zA-Z][a-zA-Z' -]{1,50})\s+hoon\b",
        r"\bmein\s+([a-zA-Z][a-zA-Z' -]{1,50})\s+hoon\b",
        r"\bname\s*[:=]\s*([a-zA-Z][a-zA-Z' -]{1,50})\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, clean_text(text), re.IGNORECASE)
        if match:
            name = clean_text(match.group(1)).strip(" .,!?")
            if name.lower() not in BAD_MEMORY_VALUES:
                return name

    return None


def get_or_create_lead(db, user_id, conversation_id):
    lead = (
        db.query(Lead)
        .filter(
            Lead.user_id == user_id,
            Lead.conversation_id == conversation_id,
        )
        .first()
    )

    if lead:
        return lead

    lead = Lead(
        user_id=user_id,
        conversation_id=conversation_id,
    )
    db.add(lead)
    db.commit()
    db.refresh(lead)
    return lead


def is_lead_interest_message(text):
    """Return True when a customer clearly expresses property interest."""
    normalized = clean_text(text).lower()

    phrases = [
        "main interested hoon",
        "mein interested hoon",
        "i am interested",
        "i'm interested",
        "interested hoon",
        "property pasand hai",
        "mujhe ye property pasand hai",
        "ye property pasand hai",
        "genuinely interested",
        "i want to buy",
        "i want to purchase",
        "khareedna hai",
        "kharidna hai",
        "meri inquiry register",
        "inquiry register kar dein",
        "inquiry register karo",
        "inquiry seller ko bhej",
        "inquiry seller ke paas bhej",
        "seller ko inquiry",
        "seller ke paas inquiry",
        "seller ko message bhej",
        "seller ke paas message bhej",
        "seller se contact karwa",
        "seller se baat karwa",
        "mujhe seller se baat",
        "contact seller",
        "send my inquiry",
    ]

    return any(phrase in normalized for phrase in phrases)


def capture_lead_from_message(db, user_id, conversation_id, message):
    text_value = clean_text(message)
    phone = extract_phone(text_value)
    name = extract_lead_name(text_value)
    interested = is_lead_interest_message(text_value)

    # Create/update a lead when contact information OR clear interest is present.
    if not phone and not name and not interested:
        return None

    lead = get_or_create_lead(
        db,
        user_id,
        conversation_id,
    )

    if name:
        lead.name = name

    if phone:
        lead.phone = phone

    saved_name = lead.name or get_memory_value(
        db,
        user_id,
        "name",
    )
    if saved_name and not lead.name:
        lead.name = clean_text(saved_name)

    # Attach the property the customer is currently discussing.
    active_property = get_active_property(
        db,
        conversation_id,
    )

    if active_property:
        lead.property_type = active_property.property_type
        lead.purpose = active_property.purpose
        lead.preferred_location = active_property.location
        lead.bedrooms = active_property.bedrooms
        lead.budget = active_property.price

        property_note = (
            f"Interested in property #{active_property.id}: "
            f"{active_property.title}."
        )

        existing_notes = clean_text(lead.notes or "")
        if property_note not in existing_notes:
            lead.notes = (
                f"{existing_notes} {property_note}"
            ).strip()

    if interested:
        lead.status = "interested"

    lead.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(lead)
    return lead


def lead_to_dict(lead):
    if not lead:
        return None

    return {
        "id": lead.id,
        "user_id": lead.user_id,
        "conversation_id": lead.conversation_id,
        "name": lead.name,
        "phone": lead.phone,
        "email": lead.email,
        "budget": lead.budget,
        "budget_lakh": money_lakh(lead.budget) if lead.budget else None,
        "preferred_location": lead.preferred_location,
        "purpose": lead.purpose,
        "property_type": lead.property_type,
        "bedrooms": lead.bedrooms,
        "notes": lead.notes,
        "status": lead.status,
        "created_at": lead.created_at,
        "updated_at": lead.updated_at,
    }


# =========================================================
# PROPERTY QUESTION DETECTORS
# =========================================================

def asks_property_search(text):

    lower = clean_text(
        text
    ).lower()

    property_words = [
        "house",
        "ghar",
        "property",
        "properties",
        "plot",
        "flat",
        "apartment",
        "home",
        "makan",
    ]

    search_words = [
        "chahiye",
        "available",
        "find",
        "search",
        "looking",
        "dhoond",
        "dhoondo",
        "dikhao",
        "dikha do",
        "show",
        "mil sakta",
        "mil sakti",
        "required",
        "need",
        "list",
        "options",
        "do",
        "hain",
    ]

    has_property_word = any(
        word in lower
        for word in property_words
    )

    has_search_word = any(
        word in lower
        for word in search_words
    )

    known_locations = [
        "bahawalpur",
        "lahore",
        "multan",
        "islamabad",
        "rawalpindi",
        "karachi",
        "faisalabad",
        "sahiwal",
        "rahim yar khan",
        "lodhran",
        "vehari",
        "khanewal",
        "pakpattan",
        "dera ghazi khan",
        "dg khan",
    ]

    has_location = any(
        location in lower
        for location in known_locations
    )

    return (
        has_property_word
        and (
            has_search_word
            or has_location
        )
    )


def asks_compare(text):

    lower = clean_text(
        text
    ).lower()

    return (
        "compare" in lower
        or "comparison" in lower
        or "dono compare" in lower
        or "dono ko compare" in lower
        or "muqabla" in lower
        or "mukabla" in lower
    )


def asks_cheaper(text):

    lower = clean_text(
        text
    ).lower()

    return (
        "sasti kaunsi" in lower
        or "sasta kaunsa" in lower
        or "cheaper" in lower
        or "cheapest" in lower
        or "kam price" in lower
        or "kam qeemat" in lower
        or "kam paisay" in lower
    )


def asks_more_bedrooms(text):

    lower = clean_text(
        text
    ).lower()

    return (
        "zyada bedrooms" in lower
        or "zyaada bedrooms" in lower
        or "more bedrooms" in lower
        or "kis mein zyada bedroom" in lower
        or "kis me zyada bedroom" in lower
    )


def asks_owner_phone(text):

    lower = clean_text(
        text
    ).lower()

    return (
        "owner ka number" in lower
        or "owner ka phone" in lower
        or "owner number" in lower
        or "contact number" in lower
        or "contact do" in lower
        or "number do" in lower
        or "phone number" in lower
    )


def asks_price(text):

    lower = clean_text(
        text
    ).lower()

    return (
        "price kya hai" in lower
        or "price batao" in lower
        or "kitne ki hai" in lower
        or "kitnay ki hai" in lower
        or "kitne ka hai" in lower
        or "kitnay ka hai" in lower
        or "cost kya hai" in lower
        or "price" == lower
    )


def asks_details(text):

    lower = clean_text(
        text
    ).lower()

    return (
        "complete details" in lower
        or "complete detail" in lower
        or "details batao" in lower
        or "detail batao" in lower
        or "poori details" in lower
        or "puri details" in lower
        or "full details" in lower
        or "iski details" in lower
        or "iske details" in lower
        or "details do" in lower
    )


# =========================================================
# NUMBERED PROPERTY SELECTION
# =========================================================

def select_numbered_property(
    db,
    conversation_id,
    message,
):

    comparison = get_comparison_properties(
        db,
        conversation_id,
    )

    if not comparison:
        return None

    lower = clean_text(
        message
    ).lower()

    selections = [
        (
            [
                "pehli",
                "pahli",
                "first",
                "1st",
            ],
            0,
        ),
        (
            [
                "doosri",
                "dusri",
                "second",
                "2nd",
            ],
            1,
        ),
        (
            [
                "teesri",
                "third",
                "3rd",
            ],
            2,
        ),
        (
            [
                "chauthi",
                "chothi",
                "fourth",
                "4th",
            ],
            3,
        ),
        (
            [
                "paanchvi",
                "panchvi",
                "fifth",
                "5th",
            ],
            4,
        ),
        (
            [
                "chhati",
                "sixth",
                "6th",
            ],
            5,
        ),
        (
            [
                "saatvi",
                "seventh",
                "7th",
            ],
            6,
        ),
        (
            [
                "aathvi",
                "eighth",
                "8th",
            ],
            7,
        ),
    ]

    for words, index in selections:

        if any(
            word in lower
            for word in words
        ):

            if len(comparison) > index:

                return comparison[index]

    # Numeric selection:
    # "property 2"
    # "option 2"
    # "2nd property"

    numeric_match = re.search(
        r"\b(?:property|option|ghar|house)\s*(\d+)\b",
        lower,
    )

    if numeric_match:

        index = (
            int(
                numeric_match.group(1)
            )
            - 1
        )

        if 0 <= index < len(comparison):

            return comparison[index]

    return None


# =========================================================
# COMPARISON RESPONSE
# =========================================================

def build_comparison(properties):

    if len(properties) < 2:

        return (
            "Comparison ke liye kam az kam "
            "2 properties chahiye."
        )

    lines = [
        "Property Comparison",
        "",
    ]

    for index, property_obj in enumerate(
        properties,
        start=1,
    ):

        lines.append(
            f"{index}. "
            f"{property_obj.title}"
        )

        lines.append(
            f"   ID: {property_obj.id}"
        )

        lines.append(
            f"   Price: "
            f"{money_lakh(property_obj.price)} lakh"
        )

        lines.append(
            f"   Size: "
            f"{property_obj.size_marla} marla"
        )

        lines.append(
            f"   Bedrooms: "
            f"{property_obj.bedrooms}"
        )

        lines.append(
            f"   Bathrooms: "
            f"{property_obj.bathrooms}"
        )

        lines.append(
            f"   Location: "
            f"{property_obj.location}"
        )

        lines.append("")

    cheapest = min(
        properties,
        key=lambda item: item.price,
    )

    expensive = max(
        properties,
        key=lambda item: item.price,
    )

    price_difference = (
        expensive.price
        - cheapest.price
    )

    lines.append(
        f"Price difference: "
        f"{money_lakh(price_difference)} lakh"
    )

    lines.append(
        f"Sasti property: "
        f"{cheapest.title}"
    )

    max_bedrooms = max(
        item.bedrooms
        for item in properties
    )

    bedroom_winners = [
        item
        for item in properties
        if item.bedrooms
        == max_bedrooms
    ]

    if len(
        bedroom_winners
    ) == len(properties):

        lines.append(
            "Sab properties mein bedrooms "
            "ki tadaad same hai."
        )

    else:

        lines.append(
            "Zyada bedrooms: "
            + ", ".join(
                item.title
                for item in bedroom_winners
            )
            + f" ({max_bedrooms} bedrooms)"
        )

    return "\n".join(lines)


# =========================================================
# ACTIVE PROPERTY QUESTIONS
# =========================================================

def answer_active_property_question(
    db,
    conversation_id,
    message,
):

    active_property = get_active_property(
        db,
        conversation_id,
    )

    if not active_property:
        return None

    lower = clean_text(
        message
    ).lower()

    if asks_owner_phone(message):

        return (
            f"{active_property.title} ke "
            f"{active_property.contact_name} ka "
            f"contact number: "
            f"{active_property.contact_phone}"
        )

    if asks_price(message):

        return (
            f"{active_property.title} ki price "
            f"{money_lakh(active_property.price)} "
            "lakh hai."
        )

    if asks_details(message):

        return format_property_details(
            active_property
        )

    if (
        "bedrooms" in lower
        or "bedroom" in lower
    ):

        return (
            f"{active_property.title} mein "
            f"{active_property.bedrooms} "
            "bedrooms hain."
        )

    if (
        "bathrooms" in lower
        or "bathroom" in lower
    ):

        return (
            f"{active_property.title} mein "
            f"{active_property.bathrooms} "
            "bathrooms hain."
        )

    if (
        "location" in lower
        or "kahan" in lower
        or "jagah" in lower
    ):

        return (
            f"{active_property.title} "
            f"{active_property.location} mein hai."
        )

    if (
        "size" in lower
        or "marla" in lower
    ):

        return (
            f"{active_property.title} "
            f"{active_property.size_marla} "
            "marla ka hai."
        )

    return None


# =========================================================
# OLLAMA
# =========================================================

def ollama_reply(
    db,
    user_id,
    conversation_id,
    message,
):

    memories = (
        db.query(Memory)
        .filter(
            Memory.user_id
            == user_id
        )
        .order_by(
            Memory.timestamp.desc()
        )
        .limit(MAX_MEMORY_ITEMS)
        .all()
    )

    memory_lines = []

    for memory in memories:

        value = clean_text(
            memory.value
        )

        if not value:
            continue

        if (
            value.lower()
            in BAD_MEMORY_VALUES
        ):
            continue

        memory_lines.append(
            f"- {memory.key}: {value}"
        )

    memory_text = (
        "\n".join(memory_lines)
        if memory_lines
        else "- No saved memory"
    )

    recent_messages = (
        db.query(Message)
        .filter(
            Message.conversation_id
            == conversation_id
        )
        .order_by(
            Message.id.desc()
        )
        .limit(MAX_HISTORY_MESSAGES)
        .all()
    )

    recent_messages.reverse()

    history_lines = []

    for item in recent_messages:

        history_lines.append(
            f"User: {item.user_message}\n"
            f"Assistant: {item.bot_reply}"
        )

    history_text = (
        "\n\n".join(history_lines)
        if history_lines
        else "No previous conversation"
    )

    active_property = get_active_property(
        db,
        conversation_id,
    )

    property_context = (
        "No active property."
    )

    if active_property:

        property_context = (
            f"ID: {active_property.id}\n"
            f"Title: {active_property.title}\n"
            f"Type: {active_property.property_type}\n"
            f"Purpose: {active_property.purpose}\n"
            f"Price: {money_lakh(active_property.price)} lakh\n"
            f"Size: {active_property.size_marla} marla\n"
            f"Bedrooms: {active_property.bedrooms}\n"
            f"Bathrooms: {active_property.bathrooms}\n"
            f"Location: {active_property.location}\n"
            f"Owner: {active_property.contact_name}\n"
            f"Phone: {active_property.contact_phone}\n"
            f"Description: {active_property.description}\n"
            f"Status: {active_property.status}"
        )

    system_prompt = """
You are an intelligent property AI assistant.

The user can speak English, Urdu, or Roman Urdu.

Rules:

1. Never invent user memory.
2. Never invent property information.
3. Never invent property IDs.
4. Never change saved facts.
5. Never change property prices.
6. Never change property locations.
7. Never change bedrooms or bathrooms.
8. Never change owner information.
9. If exact information exists in context, use it.
10. Keep answers concise and natural.
11. If property information is not provided, say you do not have that information.
12. Do not claim a property exists unless it appears in the provided context.
13. Answer in the user's language/style when possible.
"""

    user_prompt = f"""
SAVED USER MEMORY:

{memory_text}

ACTIVE PROPERTY:

{property_context}

RECENT CONVERSATION:

{history_text}

CURRENT USER MESSAGE:

{message}
"""

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "options": {
            "temperature": 0.2,
        },
    }

    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120,
        )

        response.raise_for_status()

        data = response.json()

        reply = (
            data
            .get("message", {})
            .get("content", "")
            .strip()
        )

        if reply:
            return reply

        return (
            "Mujhe response generate karne "
            "mein problem hui."
        )

    except requests.exceptions.ConnectionError:

        return (
            "Ollama se connection nahi ho raha. "
            "Check karo ke Ollama running hai aur "
            f"{OLLAMA_MODEL} model installed hai."
        )

    except requests.exceptions.Timeout:

        return (
            "Ollama response dene mein zyada "
            "time le raha hai. Dobara try karo."
        )

    except requests.exceptions.HTTPError as exc:

        print(
            "OLLAMA HTTP ERROR:",
            repr(exc),
        )

        return (
            "Ollama ne request accept nahi ki. "
            "Model aur Ollama configuration check karo."
        )

    except Exception as exc:

        print(
            "OLLAMA ERROR:",
            repr(exc),
        )

        return (
            "AI service se temporary problem "
            "aa gayi. Please dobara try karo."
        )


# =========================================================
# MAIN CHAT LOGIC
# =========================================================

def generate_reply(
    db,
    user_id,
    conversation_id,
    message,
):

    text = clean_text(
        message
    )

    # -----------------------------------------------------
    # MEMORY EXTRACTION
    # -----------------------------------------------------

    extract_memories(
        db,
        user_id,
        text,
    )

    capture_lead_from_message(
        db,
        user_id,
        conversation_id,
        text,
    )

    # -----------------------------------------------------
    # MEMORY QUESTION
    # -----------------------------------------------------

    memory_reply = answer_memory_question(
        db,
        user_id,
        text,
    )

    if memory_reply:
        return memory_reply

    # -----------------------------------------------------
    # LEAD INTEREST
    # -----------------------------------------------------
    if is_lead_interest_message(text):
        active_property = get_active_property(
            db,
            conversation_id,
        )

        lead = capture_lead_from_message(
            db,
            user_id,
            conversation_id,
            text,
        )

        if lead and active_property:
            return (
                f"Shukriya {lead.name or 'aap'}! "
                f"Aapki inquiry {active_property.title} "
                "ke liye seller/agent ke record mein save "
                f"ho gayi hai. Lead ID: {lead.id}. "
                "Seller aapse contact karega."
            )

    # -----------------------------------------------------
    # EXPLICIT PROPERTY IDs
    # -----------------------------------------------------

    explicit_ids = extract_property_ids(
        text
    )

    if explicit_ids:

        selected_properties = (
            get_properties_by_ids(
                db,
                explicit_ids,
            )
        )

        if len(
            selected_properties
        ) < 2:

            return (
                "Mujhe comparison ke liye "
                "dono valid property IDs nahi milin."
            )

        set_comparison(
            db,
            conversation_id,
            selected_properties,
        )

        set_active_property(
            db,
            conversation_id,
            selected_properties[0].id,
        )

        return build_comparison(
            selected_properties
        )

    # -----------------------------------------------------
    # EXISTING COMPARISON
    # -----------------------------------------------------

    comparison_properties = (
        get_comparison_properties(
            db,
            conversation_id,
        )
    )

    # -----------------------------------------------------
    # COMPARE
    # -----------------------------------------------------

    if asks_compare(text):

        if len(
            comparison_properties
        ) >= 2:

            return build_comparison(
                comparison_properties
            )

        return (
            "Comparison ke liye pehle "
            "kam az kam 2 properties search karo."
        )

    # -----------------------------------------------------
    # CHEAPER
    # -----------------------------------------------------

    if asks_cheaper(text):

        if len(
            comparison_properties
        ) >= 2:

            cheapest = min(
                comparison_properties,
                key=lambda item: item.price,
            )

            return (
                f"{cheapest.title} sasti hai. "
                f"Iski price "
                f"{money_lakh(cheapest.price)} "
                "lakh hai."
            )

        return (
            "Abhi comparison ke liye "
            "2 properties available nahi hain."
        )

    # -----------------------------------------------------
    # MORE BEDROOMS
    # -----------------------------------------------------

    if asks_more_bedrooms(text):

        if len(
            comparison_properties
        ) >= 2:

            max_bedrooms = max(
                item.bedrooms
                for item in comparison_properties
            )

            winners = [
                item
                for item in comparison_properties
                if item.bedrooms
                == max_bedrooms
            ]

            if len(winners) == len(
                comparison_properties
            ):

                return (
                    f"Dono properties mein "
                    f"{max_bedrooms} bedrooms hain."
                )

            return (
                "Zyada bedrooms wali property: "
                + ", ".join(
                    item.title
                    for item in winners
                )
                + f" ({max_bedrooms} bedrooms)."
            )

        return (
            "Pehle kam az kam 2 properties "
            "search karo, phir main bedrooms "
            "compare kar dunga."
        )

    # -----------------------------------------------------
    # PROPERTY SEARCH
    # -----------------------------------------------------

    if asks_property_search(text):

        properties = search_properties(
            db,
            text,
        )

        if not properties:

            return (
                "Is waqt mujhe tumhari requirements "
                "ke mutabiq koi available property "
                "nahi mili."
            )

        set_active_property(
            db,
            conversation_id,
            properties[0].id,
        )

        set_comparison(
            db,
            conversation_id,
            properties,
        )

        return format_property_list(
            properties
        )

    # -----------------------------------------------------
    # NUMBERED PROPERTY
    # -----------------------------------------------------

    selected = select_numbered_property(
        db,
        conversation_id,
        text,
    )

    if selected:

        set_active_property(
            db,
            conversation_id,
            selected.id,
        )

        if asks_owner_phone(text):

            return (
                f"{selected.title} ke "
                f"{selected.contact_name} ka "
                f"contact number: "
                f"{selected.contact_phone}"
            )

        if asks_price(text):

            return (
                f"{selected.title} ki price "
                f"{money_lakh(selected.price)} "
                "lakh hai."
            )

        if asks_details(text):

            return format_property_details(
                selected
            )

        return format_property_details(
            selected
        )

    # -----------------------------------------------------
    # ACTIVE PROPERTY
    # -----------------------------------------------------

    active_reply = (
        answer_active_property_question(
            db,
            conversation_id,
            text,
        )
    )

    if active_reply:
        return active_reply

    # -----------------------------------------------------
    # OLLAMA
    # -----------------------------------------------------

    return ollama_reply(
        db,
        user_id,
        conversation_id,
        text,
    )


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():
    db = SessionLocal()

    try:
        db.execute(text("SELECT 1"))
        return {
            "status": "ok",
            "version": APP_VERSION,
            "ai_provider": "Ollama",
            "ai_model": OLLAMA_MODEL,
            "database": "connected",
        }
    except Exception as exc:
        print("HEALTH CHECK ERROR:", repr(exc))
        raise HTTPException(
            status_code=503,
            detail="Database health check failed.",
        )
    finally:
        db.close()


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    db = SessionLocal()

    try:

        property_count = (
            db.query(Property).count()
        )

        user_count = (
            db.query(User).count()
        )

        conversation_count = (
            db.query(Conversation).count()
        )

        return {
            "message":
                "Enterprise Property AI Chatbot is running!",

            "version":
                APP_VERSION,

            "ai_model":
                OLLAMA_MODEL,

            "ai_provider":
                "Ollama",

            "database":
                (
                    "SQLite"
                    if DATABASE_URL.startswith("sqlite")
                    else "PostgreSQL/Configured SQL Database"
                ),

            "environment":
                APP_ENV,

            "authentication":
                True,

            "jwt_authentication":
                True,

            "memory":
                True,

            "conversation_context":
                True,

            "context_messages":
                10,

            "property_system":
                True,

            "active_property_context":
                True,

            "property_comparison":
                True,

            "lead_capture":
                True,

            "lead_management":
                True,

            "advanced_property_search":
                True,

            "location_filter":
                True,

            "budget_filter":
                True,

            "price_range_filter":
                True,

            "bedroom_filter":
                True,

            "bathroom_filter":
                True,

            "users":
                user_count,

            "conversations":
                conversation_count,

            "properties":
                property_count,
        }

    finally:

        db.close()


# =========================================================
# REGISTER
# =========================================================

@app.post("/auth/register")
def register(
    request: RegisterRequest,
):

    db = SessionLocal()

    try:

        name = clean_text(
            request.name
        )

        email = normalize_email(
            request.email
        )

        existing = (
            db.query(User)
            .filter(
                User.email == email
            )
            .first()
        )

        if existing:

            raise HTTPException(
                status_code=409,
                detail=(
                    "An account with this "
                    "email already exists."
                ),
            )

        user = User(
            name=name,
            email=email,
            password_hash=hash_password(
                request.password
            ),
            role="customer",
        )

        db.add(user)
        db.commit()
        db.refresh(user)

        token = create_access_token(
            user.id,
            user.role or "customer",
        )

        return {
            "message":
                "Registration successful.",

            "access_token":
                token,

            "token_type":
                "bearer",

            "expires_in_minutes":
                AUTH_TOKEN_EXPIRE_MINUTES,

            "user": {
                "id":
                    user.id,

                "name":
                    user.name,

                "email":
                    user.email,

                "role":
                    user.role or "customer",
            },
        }

    except HTTPException:

        raise

    except Exception as exc:

        db.rollback()

        print(
            "REGISTER ERROR:",
            repr(exc),
        )

        raise HTTPException(
            status_code=500,
            detail="Registration failed.",
        )

    finally:

        db.close()


# =========================================================
# AGENT REGISTRATION
# =========================================================

@app.post("/auth/register-agent")
def register_agent(
    request: AgentRegisterRequest,
):

    if not hmac.compare_digest(
        request.registration_key,
        AUTH_SECRET,
    ):
        raise HTTPException(
            status_code=403,
            detail="Invalid agent registration key.",
        )

    db = SessionLocal()

    try:

        name = clean_text(
            request.name
        )

        email = normalize_email(
            request.email
        )

        existing = (
            db.query(User)
            .filter(
                User.email == email
            )
            .first()
        )

        if existing:
            raise HTTPException(
                status_code=409,
                detail=(
                    "An account with this "
                    "email already exists."
                ),
            )

        user = User(
            name=name,
            email=email,
            password_hash=hash_password(
                request.password
            ),
            role="agent",
        )

        db.add(user)
        db.commit()
        db.refresh(user)

        token = create_access_token(
            user.id,
            user.role or "agent",
        )

        return {
            "message":
                "Agent registration successful.",

            "access_token":
                token,

            "token_type":
                "bearer",

            "expires_in_minutes":
                AUTH_TOKEN_EXPIRE_MINUTES,

            "user": {
                "id": user.id,
                "name": user.name,
                "email": user.email,
                "role": user.role,
            },
        }

    finally:

        db.close()


# =========================================================
# LOGIN
# =========================================================

@app.post("/auth/login")
def login(
    request: LoginRequest,
):

    db = SessionLocal()

    try:

        email = normalize_email(
            request.email
        )

        user = (
            db.query(User)
            .filter(
                User.email == email
            )
            .first()
        )

        if (
            not user
            or not user.password_hash
        ):

            raise HTTPException(
                status_code=401,
                detail=(
                    "Invalid email "
                    "or password."
                ),
            )

        if not verify_password(
            request.password,
            user.password_hash,
        ):

            raise HTTPException(
                status_code=401,
                detail=(
                    "Invalid email "
                    "or password."
                ),
            )

        token = create_access_token(
            user.id,
            user.role or "customer",
        )

        return {
            "message":
                "Login successful.",

            "access_token":
                token,

            "token_type":
                "bearer",

            "expires_in_minutes":
                AUTH_TOKEN_EXPIRE_MINUTES,

            "user": {
                "id":
                    user.id,

                "name":
                    user.name,

                "email":
                    user.email,

                "role":
                    user.role or "customer",
            },
        }

    finally:

        db.close()


# =========================================================
# ME
# =========================================================

@app.get("/auth/me")
def auth_me(
    current_user_id: int = Depends(
        get_current_user
    ),
):

    db = SessionLocal()

    try:

        user = (
            db.query(User)
            .filter(
                User.id
                == current_user_id
            )
            .first()
        )

        if not user:

            raise HTTPException(
                status_code=401,
                detail=(
                    "User account "
                    "no longer exists."
                ),
            )

        return {
            "id":
                user.id,

            "name":
                user.name,

            "email":
                user.email,

            "role":
                user.role or "customer",
        }

    finally:

        db.close()


# =========================================================
# CREATE USER LEGACY
# =========================================================

@app.post("/users")
def create_user(
    name: str,
    current_user_id: int = Depends(
        require_roles("admin")
    ),
):

    db = SessionLocal()

    try:

        name = clean_text(
            name
        )

        if not name:

            raise HTTPException(
                status_code=400,
                detail="Name cannot be empty.",
            )

        user = User(
            name=name,
            role="customer",
        )

        db.add(user)
        db.commit()
        db.refresh(user)

        return {
            "id":
                user.id,

            "name":
                user.name,

            "role":
                user.role,
        }

    finally:

        db.close()


# =========================================================
# GET USERS
# =========================================================

@app.get("/users")
def get_users(
    current_user_id: int = Depends(
        require_roles("admin")
    ),
):

    db = SessionLocal()

    try:

        users = (
            db.query(User)
            .order_by(
                User.id
            )
            .all()
        )

        return {
            "users": [
                {
                    "id":
                        user.id,

                    "name":
                        user.name,

                    "email":
                        user.email,

                    "role":
                        user.role or "customer",

                    "created_at":
                        user.created_at,
                }
                for user in users
            ]
        }

    finally:

        db.close()


# =========================================================
# CREATE CONVERSATION
# =========================================================

@app.post(
    "/conversations/{user_id}"
)
def create_conversation(
    user_id: int,
    current_user_id: int = Depends(
        get_current_user
    ),
):

    db = SessionLocal()

    try:

        if user_id != current_user_id:

            raise HTTPException(
                status_code=403,
                detail=(
                    "You can only create "
                    "conversations for your "
                    "own account."
                ),
            )

        user = (
            db.query(User)
            .filter(
                User.id == user_id
            )
            .first()
        )

        if not user:

            raise HTTPException(
                status_code=404,
                detail="User not found.",
            )

        conversation = Conversation(
            user_id=user_id
        )

        db.add(conversation)
        db.commit()
        db.refresh(conversation)

        return {
            "conversation_id":
                conversation.id,

            "user_id":
                user_id,

            "created_at":
                conversation.created_at,
        }

    finally:

        db.close()


# =========================================================
# CHAT
# =========================================================

@app.post(
    "/chat/{conversation_id}",
    response_model=ChatResponse,
)
def chat(
    conversation_id: int,
    request: ChatRequest,
    current_user_id: int = Depends(
        get_current_user
    ),
):

    db = SessionLocal()

    try:

        conversation = (
            db.query(Conversation)
            .filter(
                Conversation.id
                == conversation_id
            )
            .first()
        )

        if not conversation:

            raise HTTPException(
                status_code=404,
                detail="Conversation not found.",
            )

        if (
            conversation.user_id
            != current_user_id
        ):

            raise HTTPException(
                status_code=403,
                detail=(
                    "You do not have access "
                    "to this conversation."
                ),
            )

        message_text = clean_text(
            request.message
        )

        reply = generate_reply(
            db=db,
            user_id=conversation.user_id,
            conversation_id=conversation_id,
            message=message_text,
        )

        message = Message(
            conversation_id=conversation_id,
            user_message=message_text,
            bot_reply=reply,
        )

        db.add(message)
        db.commit()

        return ChatResponse(
            reply=reply
        )

    except HTTPException:

        raise

    except Exception as exc:

        db.rollback()

        print(
            "CHAT ERROR:",
            repr(exc),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Chat processing error."
            ),
        )

    finally:

        db.close()


# =========================================================
# CREATE PROPERTY
# =========================================================

@app.post("/properties")
def create_property(
    request: PropertyCreate,
    current_user_id: int = Depends(
        require_roles(
            "admin",
            "agent",
        )
    ),
):

    db = SessionLocal()

    try:

        property_obj = Property(
            title=clean_text(
                request.title
            ),

            property_type=clean_text(
                request.property_type
            ).lower(),

            purpose=clean_text(
                request.purpose
            ).lower(),

            location=clean_text(
                request.location
            ),

            price=request.price,

            size_marla=request.size_marla,

            bedrooms=request.bedrooms,

            bathrooms=request.bathrooms,

            description=clean_text(
                request.description
            ),

            status=clean_text(
                request.status
            ).lower(),

            contact_name=clean_text(
                request.contact_name
            ),

            contact_phone=clean_text(
                request.contact_phone
            ),
        )

        db.add(property_obj)
        db.commit()
        db.refresh(property_obj)

        return property_to_dict(
            property_obj
        )

    finally:

        db.close()


# =========================================================
# UPDATE PROPERTY
# =========================================================

@app.put(
    "/properties/{property_id}"
)
def update_property(
    property_id: int,
    request: PropertyUpdate,
    current_user_id: int = Depends(
        require_roles(
            "admin",
            "agent",
        )
    ),
):

    db = SessionLocal()

    try:

        property_obj = (
            db.query(Property)
            .filter(
                Property.id
                == property_id
            )
            .first()
        )

        if not property_obj:

            raise HTTPException(
                status_code=404,
                detail="Property not found.",
            )

        property_obj.title = clean_text(
            request.title
        )

        property_obj.property_type = clean_text(
            request.property_type
        ).lower()

        property_obj.purpose = clean_text(
            request.purpose
        ).lower()

        property_obj.location = clean_text(
            request.location
        )

        property_obj.price = request.price

        property_obj.size_marla = (
            request.size_marla
        )

        property_obj.bedrooms = (
            request.bedrooms
        )

        property_obj.bathrooms = (
            request.bathrooms
        )

        property_obj.description = clean_text(
            request.description
        )

        property_obj.status = clean_text(
            request.status
        ).lower()

        property_obj.contact_name = clean_text(
            request.contact_name
        )

        property_obj.contact_phone = clean_text(
            request.contact_phone
        )

        db.commit()
        db.refresh(property_obj)

        return {
            "message":
                "Property updated successfully.",

            "property":
                property_to_dict(
                    property_obj
                ),
        }

    except HTTPException:

        raise

    except Exception as exc:

        db.rollback()

        print(
            "PROPERTY UPDATE ERROR:",
            repr(exc),
        )

        raise HTTPException(
            status_code=500,
            detail="Property update failed.",
        )

    finally:

        db.close()


# =========================================================
# DELETE PROPERTY
# =========================================================

@app.delete("/properties/{property_id}")
def delete_property(
    property_id: int,
    current_user_id: int = Depends(
        require_roles("admin", "agent")
    ),
):
    db = SessionLocal()

    try:
        property_obj = (
            db.query(Property)
            .filter(Property.id == property_id)
            .first()
        )

        if not property_obj:
            raise HTTPException(
                status_code=404,
                detail="Property not found.",
            )

        # Remove conversation state that points to the deleted property.
        db.query(ActiveProperty).filter(
            ActiveProperty.property_id == property_id
        ).delete(synchronize_session=False)

        # Remove the property from saved comparison lists.
        comparisons = db.query(Comparison).all()
        for comparison in comparisons:
            ids = [
                item.strip()
                for item in (comparison.property_ids or "").split(",")
                if item.strip().isdigit()
                and int(item.strip()) != property_id
            ]
            comparison.property_ids = ",".join(ids)

        db.delete(property_obj)
        db.commit()

        return {
            "message": "Property deleted successfully.",
            "property_id": property_id,
        }

    except HTTPException:
        raise
    except Exception as exc:
        db.rollback()
        print("PROPERTY DELETE ERROR:", repr(exc))
        raise HTTPException(
            status_code=500,
            detail="Property deletion failed.",
        )
    finally:
        db.close()


# =========================================================
# GET PROPERTIES
# =========================================================

@app.get("/properties")
def get_properties():

    db = SessionLocal()

    try:

        properties = (
            db.query(Property)
            .order_by(
                Property.id
            )
            .all()
        )

        return {
            "count":
                len(properties),

            "properties": [
                property_to_dict(
                    item
                )
                for item in properties
            ],
        }

    finally:

        db.close()


# =========================================================
# GET PROPERTY
# =========================================================

@app.get(
    "/properties/{property_id}"
)
def get_property(
    property_id: int,
):

    db = SessionLocal()

    try:

        property_obj = (
            db.query(Property)
            .filter(
                Property.id
                == property_id
            )
            .first()
        )

        if not property_obj:

            raise HTTPException(
                status_code=404,
                detail="Property not found.",
            )

        return property_to_dict(
            property_obj
        )

    finally:

        db.close()


# =========================================================
# USER MEMORY CREATE / UPDATE
# =========================================================

@app.post(
    "/users/{user_id}/memory"
)
def create_user_memory(
    user_id: int,
    request: MemoryCreate,
    current_user_id: int = Depends(
        get_current_user
    ),
):

    db = SessionLocal()

    try:

        if user_id != current_user_id:

            raise HTTPException(
                status_code=403,
                detail=(
                    "You can only modify "
                    "your own memory."
                ),
            )

        user = (
            db.query(User)
            .filter(
                User.id == user_id
            )
            .first()
        )

        if not user:

            raise HTTPException(
                status_code=404,
                detail="User not found.",
            )

        memory = save_memory(
            db,
            user_id,
            request.key,
            request.value,
        )

        return {
            "message":
                "Memory saved successfully.",

            "user_id":
                user_id,

            "memory": {
                "key":
                    memory.key,

                "value":
                    memory.value,

                "timestamp":
                    memory.timestamp,
            },
        }

    finally:

        db.close()


# =========================================================
# GET USER MEMORY
# =========================================================

@app.get(
    "/users/{user_id}/memory"
)
def get_user_memory(
    user_id: int,
    current_user_id: int = Depends(
        get_current_user
    ),
):

    db = SessionLocal()

    try:

        if user_id != current_user_id:

            raise HTTPException(
                status_code=403,
                detail=(
                    "You can only access "
                    "your own memory."
                ),
            )

        user = (
            db.query(User)
            .filter(
                User.id == user_id
            )
            .first()
        )

        if not user:

            raise HTTPException(
                status_code=404,
                detail="User not found.",
            )

        memories = (
            db.query(Memory)
            .filter(
                Memory.user_id
                == user_id
            )
            .order_by(
                Memory.id
            )
            .all()
        )

        return {
            "user_id":
                user_id,

            "memory": [
                {
                    "key":
                        memory.key,

                    "value":
                        memory.value,

                    "timestamp":
                        memory.timestamp,
                }
                for memory in memories
            ],
        }

    finally:

        db.close()


# =========================================================
# CONVERSATION MESSAGES
# =========================================================

@app.get(
    "/conversations/{conversation_id}/messages"
)
def get_messages(
    conversation_id: int,
    current_user_id: int = Depends(
        get_current_user
    ),
):

    db = SessionLocal()

    try:

        conversation = (
            db.query(Conversation)
            .filter(
                Conversation.id
                == conversation_id
            )
            .first()
        )

        if not conversation:

            raise HTTPException(
                status_code=404,
                detail="Conversation not found.",
            )

        if (
            conversation.user_id
            != current_user_id
        ):

            raise HTTPException(
                status_code=403,
                detail=(
                    "You do not have access "
                    "to this conversation."
                ),
            )

        messages = (
            db.query(Message)
            .filter(
                Message.conversation_id
                == conversation_id
            )
            .order_by(
                Message.id
            )
            .all()
        )

        return {
            "conversation_id":
                conversation_id,

            "messages": [
                {
                    "id":
                        item.id,

                    "user_message":
                        item.user_message,

                    "bot_reply":
                        item.bot_reply,

                    "timestamp":
                        item.timestamp,
                }
                for item in messages
            ],
        }

    finally:

        db.close()


# =========================================================
# ACTIVE PROPERTY API
# =========================================================

@app.get(
    "/conversations/{conversation_id}/active-property"
)
def get_active_property_api(
    conversation_id: int,
    current_user_id: int = Depends(
        get_current_user
    ),
):

    db = SessionLocal()

    try:

        conversation = (
            db.query(Conversation)
            .filter(
                Conversation.id
                == conversation_id
            )
            .first()
        )

        if not conversation:

            raise HTTPException(
                status_code=404,
                detail="Conversation not found.",
            )

        if (
            conversation.user_id
            != current_user_id
        ):

            raise HTTPException(
                status_code=403,
                detail=(
                    "You do not have access "
                    "to this conversation."
                ),
            )

        property_obj = get_active_property(
            db,
            conversation_id,
        )

        return {
            "conversation_id":
                conversation_id,

            "active_property":
                property_to_dict(
                    property_obj
                ),
        }

    finally:

        db.close()


# =========================================================
# COMPARISON API
# =========================================================

@app.get(
    "/conversations/{conversation_id}/compare-properties"
)
def compare_properties_api(
    conversation_id: int,
    current_user_id: int = Depends(
        get_current_user
    ),
):

    db = SessionLocal()

    try:

        conversation = (
            db.query(Conversation)
            .filter(
                Conversation.id
                == conversation_id
            )
            .first()
        )

        if not conversation:

            raise HTTPException(
                status_code=404,
                detail="Conversation not found.",
            )

        if (
            conversation.user_id
            != current_user_id
        ):

            raise HTTPException(
                status_code=403,
                detail=(
                    "You do not have access "
                    "to this conversation."
                ),
            )

        properties = (
            get_comparison_properties(
                db,
                conversation_id,
            )
        )

        return {
            "conversation_id":
                conversation_id,

            "count":
                len(properties),

            "comparison":
                (
                    build_comparison(
                        properties
                    )
                    if len(properties) >= 2
                    else
                    "Comparison ke liye "
                    "2 properties nahi hain."
                ),

            "properties": [
                property_to_dict(
                    item
                )
                for item in properties
            ],
        }

    finally:

        db.close()


# =========================================================
# LEAD API
# =========================================================

@app.post("/conversations/{conversation_id}/lead")
def create_or_update_lead(
    conversation_id: int,
    request: LeadCreate,
    current_user_id: int = Depends(get_current_user),
):
    db = SessionLocal()

    try:
        conversation = (
            db.query(Conversation)
            .filter(Conversation.id == conversation_id)
            .first()
        )

        if not conversation:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found.",
            )

        if conversation.user_id != current_user_id:
            raise HTTPException(
                status_code=403,
                detail="You do not have access to this conversation.",
            )

        lead = get_or_create_lead(
            db,
            current_user_id,
            conversation_id,
        )

        for field in [
            "name",
            "phone",
            "email",
            "budget",
            "preferred_location",
            "purpose",
            "property_type",
            "bedrooms",
            "notes",
            "status",
        ]:
            value = getattr(request, field)
            if value is not None:
                if isinstance(value, str):
                    value = clean_text(value)
                setattr(lead, field, value)

        lead.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(lead)

        return {
            "message": "Lead saved successfully.",
            "lead": lead_to_dict(lead),
        }

    finally:
        db.close()


@app.get("/conversations/{conversation_id}/lead")
def get_conversation_lead(
    conversation_id: int,
    current_user_id: int = Depends(get_current_user),
):
    db = SessionLocal()

    try:
        conversation = (
            db.query(Conversation)
            .filter(Conversation.id == conversation_id)
            .first()
        )

        if not conversation:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found.",
            )

        if conversation.user_id != current_user_id:
            raise HTTPException(
                status_code=403,
                detail="You do not have access to this conversation.",
            )

        lead = (
            db.query(Lead)
            .filter(
                Lead.conversation_id == conversation_id,
                Lead.user_id == current_user_id,
            )
            .first()
        )

        return {
            "conversation_id": conversation_id,
            "lead": lead_to_dict(lead),
        }

    finally:
        db.close()


@app.get("/leads")
def get_leads(
    current_user_id: int = Depends(
        require_roles("admin", "agent")
    ),
):
    db = SessionLocal()

    try:
        leads = (
            db.query(Lead)
            .order_by(Lead.id.desc())
            .all()
        )

        return {
            "count": len(leads),
            "leads": [
                lead_to_dict(lead)
                for lead in leads
            ],
        }

    finally:
        db.close()


# =========================================================
# AGENT DASHBOARD
# =========================================================
@app.put("/agent/leads/{lead_id}/status")
def update_lead_status(
    lead_id: int,
    status: str,
    current_user_id: int = Depends(
        require_roles("admin", "agent")
    ),
):
    """
    Update a lead status from the agent dashboard.

    Allowed statuses:
    new -> contacted -> qualified -> interested -> closed
    """

    allowed_statuses = {
        "new",
        "contacted",
        "qualified",
        "interested",
        "closed",
    }

    if status not in allowed_statuses:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid status. Allowed statuses: "
                "new, contacted, qualified, interested, closed."
            ),
        )

    db = SessionLocal()

    try:
        lead = (
            db.query(Lead)
            .filter(Lead.id == lead_id)
            .first()
        )

        if not lead:
            raise HTTPException(
                status_code=404,
                detail="Lead not found.",
            )

        lead.status = status
        lead.updated_at = datetime.now(timezone.utc)

        db.commit()
        db.refresh(lead)

        return {
            "message": "Lead status updated successfully.",
            "lead": lead_to_dict(lead),
        }

    finally:
        db.close()


@app.get("/agent/dashboard")
def agent_dashboard(
    current_user_id: int = Depends(
        require_roles("admin", "agent")
    ),
):
    """
    Agent/admin dashboard summary for the real-estate demo.

    This endpoint intentionally reuses the existing property, lead,
    and conversation tables so it does not disturb the verified
    customer-to-lead workflow.
    """

    db = SessionLocal()

    try:
        user = (
            db.query(User)
            .filter(User.id == current_user_id)
            .first()
        )

        total_properties = db.query(Property).count()

        available_properties = (
            db.query(Property)
            .filter(Property.status == "available")
            .count()
        )

        total_leads = db.query(Lead).count()

        interested_leads = (
            db.query(Lead)
            .filter(Lead.status == "interested")
            .count()
        )

        new_leads = (
            db.query(Lead)
            .filter(Lead.status == "new")
            .count()
        )

        total_conversations = db.query(Conversation).count()

        recent_leads = (
            db.query(Lead)
            .order_by(Lead.updated_at.desc())
            .limit(10)
            .all()
        )

        recent_properties = (
            db.query(Property)
            .order_by(Property.created_at.desc())
            .limit(10)
            .all()
        )

        return {
            "agent": {
                "id": current_user_id,
                "name": user.name if user else None,
                "email": user.email if user else None,
                "role": user.role if user else None,
            },
            "summary": {
                "total_properties": total_properties,
                "available_properties": available_properties,
                "total_leads": total_leads,
                "interested_leads": interested_leads,
                "new_leads": new_leads,
                "total_conversations": total_conversations,
            },
            "recent_leads": [
                lead_to_dict(lead)
                for lead in recent_leads
            ],
            "recent_properties": [
                property_to_dict(property_obj)
                for property_obj in recent_properties
            ],
        }

    finally:
        db.close()


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8002,
        reload=False,
    )