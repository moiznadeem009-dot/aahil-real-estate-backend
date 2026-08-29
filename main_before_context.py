
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
import ollama


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Enterprise Property AI Chatbot",
    version="3.0.0",
)


# ============================================================
# DATABASE
# ============================================================

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


# ============================================================
# USER
# ============================================================

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


# ============================================================
# CONVERSATION
# ============================================================

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


# ============================================================
# MESSAGE
# ============================================================

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


# ============================================================
# MEMORY
# ============================================================

class Memory(Base):
    __tablename__ = "memories"

    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "key",
            name="uq_user_memory_key",
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


# ============================================================
# PROPERTY
# ============================================================

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
    )

    purpose = Column(
        String(50),
        nullable=False,
    )

    location = Column(
        String(150),
        nullable=False,
    )

    price = Column(
        Integer,
        nullable=False,
    )

    size_marla = Column(
        Integer,
        nullable=True,
    )

    bedrooms = Column(
        Integer,
        nullable=True,
    )

    bathrooms = Column(
        Integer,
        nullable=True,
    )

    description = Column(
        Text,
        nullable=True,
    )

    status = Column(
        String(50),
        default="available",
    )

    contact_name = Column(
        String(100),
        nullable=True,
    )

    contact_phone = Column(
        String(50),
        nullable=True,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )


# ============================================================
# DATABASE SETUP
# ============================================================

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

    finally:
        db.close()


prepare_memory_database()


# ============================================================
# PYDANTIC MODELS
# ============================================================

class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=5000,
    )


class ChatResponse(BaseModel):
    reply: str


class PropertyCreate(BaseModel):
    title: str
    property_type: str
    purpose: str
    location: str
    price: int
    size_marla: int | None = None
    bedrooms: int | None = None
    bathrooms: int | None = None
    description: str | None = None
    status: str = "available"
    contact_name: str | None = None
    contact_phone: str | None = None


# ============================================================
# MEMORY HELPERS
# ============================================================

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


# ============================================================
# MEMORY EXTRACTION
# ============================================================

def extract_memories(
    db,
    user_id: int,
    message: str,
):
    text_value = message.strip()

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
            save_memory(
                db,
                user_id,
                "name",
                match.group(1).strip(),
            )
            break

    learning_patterns = [
        r"\bmain\s+(.+?)\s+seekh raha hoon\b",
        r"\bmain\s+(.+?)\s+seekh\s+raha\s+hoon\b",
        r"\bi am learning\s+(.+)",
        r"\bi'm learning\s+(.+)",
    ]

    for pattern in learning_patterns:
        match = re.search(
            pattern,
            text_value,
            re.IGNORECASE,
        )

        if match:
            value = match.group(1).strip()
            save_memory(
                db,
                user_id,
                "learning",
                value,
            )
            break

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
            save_memory(
                db,
                user_id,
                "favorite_color",
                match.group(1).strip().lower(),
            )
            break


# ============================================================
# MEMORY QUESTIONS
# ============================================================

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

    asks_learning = (
        "main kya seekh raha hoon" in lower
        or "main kya seekh raha hun" in lower
        or "what am i learning" in lower
    )

    asks_color = (
        "mera favorite color kya hai" in lower
        or "mera favourite color kya hai" in lower
        or "mera favourite colour kya hai" in lower
        or "mera favorite colour kya hai" in lower
        or "what is my favorite color" in lower
        or "what is my favourite color" in lower
    )

    if not asks_name and not asks_learning and not asks_color:
        return None

    answers = []

    if asks_name:
        name = get_memory(
            db,
            user_id,
            "name",
        )

        if name:
            answers.append(
                f"Tumhara naam {name} hai."
            )
        else:
            answers.append(
                "Tumne abhi mujhe apna naam nahi bataya."
            )

    if asks_learning:
        learning = get_memory(
            db,
            user_id,
            "learning",
        )

        if learning:
            answers.append(
                f"Tum {learning} seekh rahe ho."
            )
        else:
            answers.append(
                "Tumne abhi mujhe nahi bataya ke tum kya seekh rahe ho."
            )

    if asks_color:
        color = get_memory(
            db,
            user_id,
            "favorite_color",
        )

        if color:
            answers.append(
                f"Tumhara favorite color {color} hai."
            )
        else:
            answers.append(
                "Tumne abhi mujhe apna favorite color nahi bataya."
            )

    return " ".join(answers)


# ============================================================
# PROPERTY HELPERS
# ============================================================

def property_to_dict(property_obj):
    return {
        "id": property_obj.id,
        "title": property_obj.title,
        "property_type": property_obj.property_type,
        "purpose": property_obj.purpose,
        "location": property_obj.location,
        "price": property_obj.price,
        "size_marla": property_obj.size_marla,
        "bedrooms": property_obj.bedrooms,
        "bathrooms": property_obj.bathrooms,
        "description": property_obj.description,
        "status": property_obj.status,
        "contact_name": property_obj.contact_name,
        "contact_phone": property_obj.contact_phone,
        "created_at": property_obj.created_at,
    }


def format_lakh(price):
    return f"{price / 100000:.1f} lakh"


# ============================================================
# PROPERTY SEARCH
# ============================================================

def search_properties(
    db,
    location=None,
    property_type=None,
    purpose=None,
    min_price=None,
    max_price=None,
    min_size_marla=None,
    max_size_marla=None,
    bedrooms=None,
    bathrooms=None,
):
    query = db.query(Property)

    if location:
        query = query.filter(
            Property.location.ilike(
                f"%{location}%"
            )
        )

    if property_type:
        query = query.filter(
            Property.property_type.ilike(
                f"%{property_type}%"
            )
        )

    if purpose:
        query = query.filter(
            Property.purpose.ilike(
                f"%{purpose}%"
            )
        )

    if min_price is not None:
        query = query.filter(
            Property.price >= min_price
        )

    if max_price is not None:
        query = query.filter(
            Property.price <= max_price
        )

    if min_size_marla is not None:
        query = query.filter(
            Property.size_marla >= min_size_marla
        )

    if max_size_marla is not None:
        query = query.filter(
            Property.size_marla <= max_size_marla
        )

    if bedrooms is not None:
        query = query.filter(
            Property.bedrooms == bedrooms
        )

    if bathrooms is not None:
        query = query.filter(
            Property.bathrooms == bathrooms
        )

    query = query.filter(
        Property.status.ilike("available")
    )

    return query.order_by(Property.price.asc()).all()


# ============================================================
# NATURAL LANGUAGE PROPERTY SEARCH
# ============================================================

def parse_property_query(message):
    lower = message.lower()

    location = None

    known_locations = [
        "bahawalpur",
        "lahore",
        "islamabad",
        "rawalpindi",
        "multan",
        "karachi",
        "faisalabad",
        "peshawar",
        "quetta",
        "sialkot",
        "gujranwala",
    ]

    for city in known_locations:
        if city in lower:
            location = city
            break

    property_type = None

    if "house" in lower or "ghar" in lower:
        property_type = "house"
    elif "plot" in lower:
        property_type = "plot"
    elif "apartment" in lower or "flat" in lower:
        property_type = "apartment"
    elif "shop" in lower:
        property_type = "shop"

    purpose = None

    if "sale" in lower or "buy" in lower or "kharid" in lower:
        purpose = "sale"
    elif "rent" in lower or "rental" in lower or "kiraya" in lower:
        purpose = "rent"

    size_marla = None

    match = re.search(
        r"(\d+(?:\.\d+)?)\s*marla",
        lower,
    )

    if match:
        size_marla = int(float(match.group(1)))

    max_price = None

    # 70 lakh / 70 lac / 70 lakhs
    lakh_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:lakh|lakhs|lac)",
        lower,
    )

    if lakh_match:
        max_price = int(
            float(lakh_match.group(1)) * 100000
        )

    # 70 lakh tak
    # 7000000 tak
    if max_price is None:
        number_match = re.search(
            r"(\d[\d,]*)\s*(?:tak|maximum|max)",
            lower,
        )

        if number_match:
            number = number_match.group(1).replace(
                ",",
                "",
            )

            try:
                max_price = int(number)
            except ValueError:
                max_price = None

    bedrooms = None

    bedroom_match = re.search(
        r"(\d+)\s*(?:bed|bedroom|bedrooms)",
        lower,
    )

    if bedroom_match:
        bedrooms = int(
            bedroom_match.group(1)
        )

    bathrooms = None

    bathroom_match = re.search(
        r"(\d+)\s*(?:bath|bathroom|bathrooms)",
        lower,
    )

    if bathroom_match:
        bathrooms = int(
            bathroom_match.group(1)
        )

    return {
        "location": location,
        "property_type": property_type,
        "purpose": purpose,
        "min_price": None,
        "max_price": max_price,
        "min_size_marla": size_marla,
        "max_size_marla": size_marla,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
    }


# ============================================================
# PROPERTY RESPONSE
# ============================================================

def build_property_response(properties):
    if not properties:
        return (
            "Mujhe is waqt aapki requirements ke mutabiq "
            "koi available property nahi mili."
        )

    if len(properties) == 1:
        p = properties[0]

        return (
            f"Mujhe 1 matching property mili:\n\n"
            f"1. {p.title}\n"
            f"Price: {format_lakh(p.price)}\n"
            f"Location: {p.location}\n"
            f"Size: {p.size_marla or '-'} marla\n"
            f"Bedrooms: {p.bedrooms or '-'}\n"
            f"Bathrooms: {p.bathrooms or '-'}\n"
            f"Description: {p.description or '-'}\n\n"
            f"Property ID: {p.id}"
        )

    lines = [
        f"Mujhe {len(properties)} matching properties mili hain:\n"
    ]

    for index, p in enumerate(properties, start=1):
        lines.append(
            f"{index}. {p.title}\n"
            f"   Price: {format_lakh(p.price)}\n"
            f"   Location: {p.location}\n"
            f"   Size: {p.size_marla or '-'} marla\n"
            f"   Bedrooms: {p.bedrooms or '-'}\n"
            f"   Bathrooms: {p.bathrooms or '-'}\n"
            f"   Property ID: {p.id}\n"
        )

    lines.append(
        "Aap number bata sakte hain, jaise "
        "\"pehli wali\" ya \"2 wali\", "
        "aur main us property ki complete details bata dunga."
    )

    return "\n".join(lines)


# ============================================================
# PROPERTY SELECTION
# ============================================================

def get_selected_property(
    db,
    conversation_id,
    message,
):
    lower = message.lower().strip()

    number = None

    # 1 wali / 2 wali / number 1
    patterns = [
        r"\b(\d+)\s*(?:wali|wala|property)\b",
        r"\bnumber\s*(\d+)\b",
        r"\bno\.?\s*(\d+)\b",
        r"^(\d+)$",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            lower,
        )

        if match:
            number = int(match.group(1))
            break

    ordinal_map = {
        "pehli": 1,
        "pehla": 1,
        "dusri": 2,
        "doosri": 2,
        "dusra": 2,
        "doosra": 2,
        "teesri": 3,
        "teesra": 3,
        "chauthi": 4,
        "chautha": 4,
        "paanchvi": 5,
        "panchvi": 5,
    }

    if number is None:
        for word, value in ordinal_map.items():
            if word in lower:
                number = value
                break

    if number is None:
        return None

    # Get recent assistant message.
    db_message = (
        db.query(Message)
        .filter(
            Message.conversation_id == conversation_id,
        )
        .order_by(Message.id.desc())
        .first()
    )

    if not db_message:
        return None

    # Find property IDs in previous response.
    property_ids = re.findall(
        r"Property ID:\s*(\d+)",
        db_message.bot_reply,
        re.IGNORECASE,
    )

    if not property_ids:
        return None

    if number < 1 or number > len(property_ids):
        return None

    property_id = int(
        property_ids[number - 1]
    )

    return (
        db.query(Property)
        .filter(
            Property.id == property_id,
        )
        .first()
    )


def property_detail_response(property_obj):
    if not property_obj:
        return None

    return (
        f"{property_obj.title}\n\n"
        f"Price: {format_lakh(property_obj.price)}\n"
        f"Location: {property_obj.location}\n"
        f"Type: {property_obj.property_type}\n"
        f"Purpose: {property_obj.purpose}\n"
        f"Size: {property_obj.size_marla or '-'} marla\n"
        f"Bedrooms: {property_obj.bedrooms or '-'}\n"
        f"Bathrooms: {property_obj.bathrooms or '-'}\n"
        f"Status: {property_obj.status}\n"
        f"Description: {property_obj.description or '-'}\n"
        f"Contact: {property_obj.contact_name or '-'}\n"
        f"Phone: {property_obj.contact_phone or '-'}\n"
        f"Property ID: {property_obj.id}"
    )


# ============================================================
# OLLAMA
# ============================================================

OLLAMA_MODEL = "llama3.2:3b"


def ask_ollama(
    user_message,
    context="",
):
    system_prompt = """
Tum ek Pakistani real-estate chatbot ho.

Rules:
1. Roman Urdu mein jawab do.
2. Agar user English mein pooche to simple Roman Urdu mein jawab do.
3. Short aur natural jawab do.
4. Jo information context mein nahi hai usay invent mat karo.
5. Property information ke liye sirf provided database context use karo.
6. Prices ko lakh mein clearly explain karo.
7. Agar multiple properties hon to options clearly compare karo.
"""

    if context:
        system_prompt += (
            "\n\nDATABASE CONTEXT:\n"
            + context
        )

    try:
        result = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_message,
                },
            ],
        )

        return result["message"]["content"].strip()

    except Exception:
        return (
            "AI model se connection nahi ho saka. "
            "Please check karein ke Ollama running hai."
        )


# ============================================================
# GENERAL CHATBOT LOGIC
# ============================================================

def generate_reply(
    db,
    user_id,
    conversation_id,
    message,
):
    text_value = message.strip()

    # --------------------------------------------------------
    # MEMORY
    # --------------------------------------------------------

    extract_memories(
        db,
        user_id,
        text_value,
    )

    memory_reply = answer_memory_question(
        db,
        user_id,
        text_value,
    )

    if memory_reply:
        return memory_reply

    # --------------------------------------------------------
    # PROPERTY SELECTION
    # --------------------------------------------------------

    selected_property = get_selected_property(
        db,
        conversation_id,
        text_value,
    )

    if selected_property:
        return property_detail_response(
            selected_property
        )

    # --------------------------------------------------------
    # PROPERTY SEARCH
    # --------------------------------------------------------

    property_keywords = [
        "house",
        "ghar",
        "plot",
        "apartment",
        "flat",
        "shop",
        "property",
        "marla",
        "lakh",
        "lac",
        "sale",
        "rent",
        "kharid",
        "kiraya",
    ]

    lower = text_value.lower()

    looks_like_property_query = any(
        keyword in lower
        for keyword in property_keywords
    )

    if looks_like_property_query:
        filters = parse_property_query(
            text_value
        )

        properties = search_properties(
            db,
            **filters,
        )

        if properties:
            return build_property_response(
                properties[:10]
            )

    # --------------------------------------------------------
    # GENERAL AI CHAT
    # --------------------------------------------------------

    name = get_memory(
        db,
        user_id,
        "name",
    )

    learning = get_memory(
        db,
        user_id,
        "learning",
    )

    context = ""

    if name:
        context += f"User name: {name}\n"

    if learning:
        context += f"User is learning: {learning}\n"

    return ask_ollama(
        text_value,
        context,
    )


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Enterprise Property AI Chatbot is running!",
        "version": "3.0.0",
        "ai_model": OLLAMA_MODEL,
        "ai_provider": "Ollama",
        "conversation_context": True,
        "memory": True,
        "property_system": True,
        "database": "SQLite",
    }


# ============================================================
# CREATE USER
# ============================================================

@app.post("/users")
def create_user(name: str):
    db = SessionLocal()

    try:
        user = User(
            name=name
        )

        db.add(user)
        db.commit()
        db.refresh(user)

        return {
            "id": user.id,
            "name": user.name,
        }

    finally:
        db.close()


# ============================================================
# GET USERS
# ============================================================

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


# ============================================================
# CREATE CONVERSATION
# ============================================================

@app.post("/conversations/{user_id}")
def create_conversation(user_id: int):
    db = SessionLocal()

    try:
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


# ============================================================
# CHAT
# ============================================================

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

        reply = generate_reply(
            db,
            conversation.user_id,
            conversation_id,
            request.message,
        )

        message = Message(
            conversation_id=conversation_id,
            user_message=request.message,
            bot_reply=reply,
        )

        db.add(message)
        db.commit()

        return ChatResponse(
            reply=reply
        )

    finally:
        db.close()


# ============================================================
# CREATE PROPERTY
# ============================================================

@app.post("/properties")
def create_property(
    request: PropertyCreate,
):
    db = SessionLocal()

    try:
        property_obj = Property(
            title=request.title,
            property_type=request.property_type,
            purpose=request.purpose,
            location=request.location,
            price=request.price,
            size_marla=request.size_marla,
            bedrooms=request.bedrooms,
            bathrooms=request.bathrooms,
            description=request.description,
            status=request.status,
            contact_name=request.contact_name,
            contact_phone=request.contact_phone,
        )

        db.add(property_obj)
        db.commit()
        db.refresh(property_obj)

        return property_to_dict(
            property_obj
        )

    finally:
        db.close()


# ============================================================
# GET ALL PROPERTIES
# ============================================================

@app.get("/properties")
def get_properties():
    db = SessionLocal()

    try:
        properties = (
            db.query(Property)
            .order_by(Property.id)
            .all()
        )

        return {
            "count": len(properties),
            "properties": [
                property_to_dict(p)
                for p in properties
            ],
        }

    finally:
        db.close()


# ============================================================
# GET SINGLE PROPERTY
# ============================================================

@app.get("/properties/{property_id}")
def get_property(
    property_id: int,
):
    db = SessionLocal()

    try:
        property_obj = (
            db.query(Property)
            .filter(
                Property.id == property_id
            )
            .first()
        )

        if not property_obj:
            raise HTTPException(
                status_code=404,
                detail="Property not found",
            )

        return property_to_dict(
            property_obj
        )

    finally:
        db.close()


# ============================================================
# PROPERTY SEARCH
# IMPORTANT:
# This route is BEFORE /properties/{property_id}
# so /search is not interpreted as an integer ID.
# ============================================================

@app.get("/properties/search")
def search_properties_api(
    location: str | None = None,
    property_type: str | None = None,
    purpose: str | None = None,
    min_price: int | None = None,
    max_price: int | None = None,
    min_size_marla: int | None = None,
    max_size_marla: int | None = None,
    bedrooms: int | None = None,
    bathrooms: int | None = None,
):
    db = SessionLocal()

    try:
        properties = search_properties(
            db,
            location=location,
            property_type=property_type,
            purpose=purpose,
            min_price=min_price,
            max_price=max_price,
            min_size_marla=min_size_marla,
            max_size_marla=max_size_marla,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
        )

        return {
            "count": len(properties),
            "filters": {
                "location": location,
                "property_type": property_type,
                "purpose": purpose,
                "min_price": min_price,
                "max_price": max_price,
                "min_size_marla": min_size_marla,
                "max_size_marla": max_size_marla,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
            },
            "properties": [
                property_to_dict(p)
                for p in properties
            ],
        }

    finally:
        db.close()


# ============================================================
# CONVERSATION HISTORY
# ============================================================

@app.get(
    "/conversations/{conversation_id}/messages"
)
def get_messages(
    conversation_id: int,
):
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
                Message.conversation_id
                == conversation_id,
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


# ============================================================
# USER MEMORY
# ============================================================

@app.get("/users/{user_id}/memory")
def get_user_memory(
    user_id: int,
):
    db = SessionLocal()

    try:
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
