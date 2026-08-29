from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
import re


app = FastAPI(title="Enterprise Chatbot API")


# ==================================================
# DATABASE
# ==================================================

DATABASE_URL = "sqlite:///./chatbot.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


# ==================================================
# USER
# ==================================================

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)

    conversations = relationship(
        "Conversation",
        back_populates="user",
    )

    memories = relationship(
        "Memory",
        back_populates="user",
    )


# ==================================================
# CONVERSATION
# ==================================================

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=False,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )

    user = relationship(
        "User",
        back_populates="conversations",
    )

    messages = relationship(
        "Message",
        back_populates="conversation",
    )


# ==================================================
# MESSAGE
# ==================================================

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)

    conversation_id = Column(
        Integer,
        ForeignKey("conversations.id"),
        nullable=False,
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
        default=datetime.utcnow,
    )

    conversation = relationship(
        "Conversation",
        back_populates="messages",
    )


# ==================================================
# MEMORY
# ==================================================

class Memory(Base):
    __tablename__ = "memories"

    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "key",
            name="uq_user_memory_key",
        ),
    )

    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=False,
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
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    user = relationship(
        "User",
        back_populates="memories",
    )


# ==================================================
# DATABASE SETUP
# ==================================================

Base.metadata.create_all(bind=engine)


def prepare_memory_database():
    db = SessionLocal()

    try:
        duplicate_groups = db.execute(
            text(
                """
                SELECT user_id, key, MAX(id) AS keep_id
                FROM memories
                GROUP BY user_id, key
                HAVING COUNT(*) > 1
                """
            )
        ).fetchall()

        for user_id, key, keep_id in duplicate_groups:
            db.execute(
                text(
                    """
                    DELETE FROM memories
                    WHERE user_id = :user_id
                      AND key = :key
                      AND id != :keep_id
                    """
                ),
                {
                    "user_id": user_id,
                    "key": key,
                    "keep_id": keep_id,
                },
            )

        db.commit()

        db.execute(
            text(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS
                uq_user_memory_key
                ON memories (user_id, key)
                """
            )
        )

        db.commit()

    finally:
        db.close()


prepare_memory_database()


# ==================================================
# PYDANTIC MODELS
# ==================================================

class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=5000,
    )


class ChatResponse(BaseModel):
    reply: str


# ==================================================
# MEMORY HELPERS
# ==================================================

def save_memory(
    db,
    user_id: int,
    key: str,
    value: str,
):
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
        memory.timestamp = datetime.utcnow()
    else:
        db.add(
            Memory(
                user_id=user_id,
                key=key,
                value=value,
            )
        )

    db.commit()


def get_memory(
    db,
    user_id: int,
    key: str,
):
    memory = (
        db.query(Memory)
        .filter(
            Memory.user_id == user_id,
            Memory.key == key,
        )
        .first()
    )

    return memory.value if memory else None


# ==================================================
# EXTRACT MEMORY
# ==================================================

def extract_memories(db, user_id: int, message: str):
    """
    English + Roman Urdu memory extraction.

    Examples:
    - Mera naam Aahil hai
    - My name is Aahil
    - Hello, my name is Aahil
    - Mera favorite color blue hai
    - My favorite color is blue
    - My favourite colour is blue
    """

    text_value = message.strip()
    lower = text_value.lower()

    # ----------------------------------------------
    # NAME
    # ----------------------------------------------

    name_patterns = [
        r"\bmera naam\s+([a-zA-Z][a-zA-Z'-]*)\s+hai\b",
        r"\bmera name\s+([a-zA-Z][a-zA-Z'-]*)\s+hai\b",
        r"\bmy name is\s+([a-zA-Z][a-zA-Z'-]*)\b",
        r"\bname is\s+([a-zA-Z][a-zA-Z'-]*)\b",
    ]

    for pattern in name_patterns:
        match = re.search(
            pattern,
            text_value,
            re.IGNORECASE,
        )

        if match:
            name = match.group(1).strip()
            save_memory(
                db,
                user_id,
                "name",
                name,
            )
            break

    # ----------------------------------------------
    # FAVORITE COLOR
    # ----------------------------------------------

    color_patterns = [
        r"\b(?:my\s+)?favou?rite\s+colou?r\s+(?:is|hai|=)\s*([a-zA-Z]+)\b",
        r"\b(?:mera\s+)?favou?rite\s+colou?r\s+([a-zA-Z]+)\s+hai\b",
    ]

    for pattern in color_patterns:
        match = re.search(
            pattern,
            text_value,
            re.IGNORECASE,
        )

        if match:
            color = match.group(1).strip().lower()
            save_memory(
                db,
                user_id,
                "favorite_color",
                color,
            )
            break


# ==================================================
# ANSWER MEMORY QUESTIONS
# ==================================================

def answer_memory_question(
    db,
    user_id: int,
    message: str,
):
    lower = message.lower()

    asks_name = (
        "mera naam kya hai" in lower
        or "mera name kya hai" in lower
        or "what is my name" in lower
        or "what's my name" in lower
        or "whats my name" in lower
    )

    asks_color = (
        "mera favorite color kya hai" in lower
        or "mera favourite color kya hai" in lower
        or "mera favourite colour kya hai" in lower
        or "mera favorite colour kya hai" in lower
        or "what is my favorite color" in lower
        or "what is my favourite color" in lower
        or "what's my favorite color" in lower
        or "what's my favourite color" in lower
        or "whats my favorite color" in lower
        or "whats my favourite color" in lower
    )

    if not asks_name and not asks_color:
        return None

    name = get_memory(db, user_id, "name") if asks_name else None
    color = (
        get_memory(db, user_id, "favorite_color")
        if asks_color
        else None
    )

    answers = []

    if asks_name:
        if name:
            answers.append(f"Tumhara naam {name} hai.")
        else:
            answers.append(
                "Tumne abhi mujhe apna naam nahi bataya."
            )

    if asks_color:
        if color:
            answers.append(
                f"Tumhara favorite color {color} hai."
            )
        else:
            answers.append(
                "Tumne abhi mujhe apna favorite color nahi bataya."
            )

    return " ".join(answers)


# ==================================================
# CHATBOT LOGIC
# ==================================================

def generate_reply(
    db,
    user_id: int,
    message: str,
):
    text_value = message.strip()

    # Save any new memories first.
    extract_memories(
        db,
        user_id,
        text_value,
    )

    # Answer memory questions if this is a memory question.
    memory_reply = answer_memory_question(
        db,
        user_id,
        text_value,
    )

    if memory_reply:
        return memory_reply

    # Friendly confirmation when memory was detected.
    lower = text_value.lower()

    saved_name = get_memory(db, user_id, "name")
    saved_color = get_memory(db, user_id, "favorite_color")

    name_was_given = (
        "mera naam " in lower
        or "mera name " in lower
        or "my name is " in lower
        or "name is " in lower
    )

    color_was_given = (
        "favorite color" in lower
        or "favourite color" in lower
        or "favorite colour" in lower
        or "favourite colour" in lower
        or "favourite colour" in lower
    )

    if name_was_given and saved_name:
        return (
            f"Theek hai, main yaad rakhunga "
            f"ke tumhara naam {saved_name} hai."
        )

    if color_was_given and saved_color:
        return (
            f"Theek hai, main yaad rakhunga "
            f"ke tumhara favorite color {saved_color} hai."
        )

    return f"You said: {text_value}"


# ==================================================
# HOME
# ==================================================

@app.get("/")
def home():
    return {
        "message": "Enterprise Chatbot API is running!"
    }


# ==================================================
# CREATE USER
# ==================================================

@app.post("/users")
def create_user(name: str):
    db = SessionLocal()

    try:
        user = User(name=name)

        db.add(user)
        db.commit()
        db.refresh(user)

        return {
            "id": user.id,
            "name": user.name,
        }

    finally:
        db.close()


# ==================================================
# GET USERS
# ==================================================

@app.get("/users")
def get_users():
    db = SessionLocal()

    try:
        users = db.query(User).all()

        return {
            "users": [
                {
                    "id": user.id,
                    "name": user.name,
                }
                for user in users
            ]
        }

    finally:
        db.close()


# ==================================================
# CREATE CONVERSATION
# ==================================================

@app.post("/conversations/{user_id}")
def create_conversation(user_id: int):
    db = SessionLocal()

    try:
        user = (
            db.query(User)
            .filter(User.id == user_id)
            .first()
        )

        if not user:
            raise HTTPException(
                status_code=404,
                detail="User not found",
            )

        conversation = Conversation(
            user_id=user_id,
        )

        db.add(conversation)
        db.commit()
        db.refresh(conversation)

        return {
            "conversation_id": conversation.id,
            "user_id": user_id,
        }

    finally:
        db.close()


# ==================================================
# CHAT
# ==================================================

@app.post(
    "/chat/{conversation_id}",
    response_model=ChatResponse,
)
def chat(
    conversation_id: int,
    request: ChatRequest,
):
    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty",
        )

    db = SessionLocal()

    try:
        conversation = (
            db.query(Conversation)
            .filter(
                Conversation.id == conversation_id,
            )
            .first()
        )

        if not conversation:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found",
            )

        user_id = conversation.user_id

        reply = generate_reply(
            db,
            user_id,
            request.message,
        )

        message = Message(
            conversation_id=conversation_id,
            user_message=request.message,
            bot_reply=reply,
        )

        db.add(message)
        db.commit()

        return ChatResponse(reply=reply)

    finally:
        db.close()


# ==================================================
# CONVERSATION HISTORY
# ==================================================

@app.get(
    "/conversations/{conversation_id}/messages"
)
def get_messages(conversation_id: int):
    db = SessionLocal()

    try:
        conversation = (
            db.query(Conversation)
            .filter(
                Conversation.id == conversation_id,
            )
            .first()
        )

        if not conversation:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found",
            )

        messages = (
            db.query(Message)
            .filter(
                Message.conversation_id == conversation_id,
            )
            .order_by(Message.id)
            .all()
        )

        return {
            "conversation_id": conversation_id,
            "messages": [
                {
                    "id": message.id,
                    "user_message": message.user_message,
                    "bot_reply": message.bot_reply,
                    "timestamp": message.timestamp,
                }
                for message in messages
            ],
        }

    finally:
        db.close()


# ==================================================
# USER MEMORY
# ==================================================

@app.get("/users/{user_id}/memory")
def get_user_memory(user_id: int):
    db = SessionLocal()

    try:
        user = (
            db.query(User)
            .filter(User.id == user_id)
            .first()
        )

        if not user:
            raise HTTPException(
                status_code=404,
                detail="User not found",
            )

        memories = (
            db.query(Memory)
            .filter(
                Memory.user_id == user_id,
            )
            .order_by(Memory.id)
            .all()
        )

        return {
            "user_id": user_id,
            "memory": [
                {
                    "key": memory.key,
                    "value": memory.value,
                    "timestamp": memory.timestamp,
                }
                for memory in memories
            ],
        }

    finally:
        db.close()