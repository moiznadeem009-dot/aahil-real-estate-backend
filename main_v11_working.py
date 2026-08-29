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
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
import requests
import re


# =========================================================
# CONFIGURATION
# =========================================================

APP_VERSION = "11.0.0"

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "llama3.2:3b"

DATABASE_URL = "sqlite:///./chatbot.db"


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="Enterprise Property AI Chatbot",
    version=APP_VERSION,
)


# =========================================================
# DATABASE
# =========================================================

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


# =========================================================
# USER
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

    conversations = relationship(
        "Conversation",
        back_populates="user",
    )

    memories = relationship(
        "Memory",
        back_populates="user",
    )


# =========================================================
# CONVERSATION
# =========================================================

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


# =========================================================
# MESSAGE
# =========================================================

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


# =========================================================
# MEMORY
# =========================================================

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


# =========================================================
# PROPERTY
# =========================================================

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
        default=datetime.utcnow,
    )


# =========================================================
# ACTIVE PROPERTY
# =========================================================

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
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )


# =========================================================
# COMPARISON
# =========================================================

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
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
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


class PropertyCreate(BaseModel):
    title: str
    property_type: str = "house"
    purpose: str = "sale"
    location: str
    price: int
    size_marla: int
    bedrooms: int
    bathrooms: int
    description: str = ""
    status: str = "available"
    contact_name: str
    contact_phone: str


# =========================================================
# BASIC HELPERS
# =========================================================

def clean_text(value):
    if value is None:
        return ""

    return str(value).strip()


def money_lakh(price):
    try:
        return round(float(price) / 100000, 2)
    except Exception:
        return 0


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


# =========================================================
# MEMORY FUNCTIONS
# =========================================================

BAD_MEMORY_VALUES = {
    "kya",
    "hai",
    "hain",
    "ka",
    "ki",
    "ke",
    "mera",
    "naam",
    "name",
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
    value = clean_text(value)

    if not value:
        return

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
        memory = Memory(
            user_id=user_id,
            key=key,
            value=value,
        )

        db.add(memory)

    db.commit()


def get_memory(
    db,
    user_id,
    key,
):
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

    value = clean_text(memory.value)

    if value.lower() in BAD_MEMORY_VALUES:
        return None

    return value


# =========================================================
# MEMORY CLEANUP
# =========================================================

def cleanup_bad_memory_values(db):
    memories = db.query(Memory).all()

    changed = False

    for memory in memories:
        value = clean_text(memory.value)

        if value.lower() in BAD_MEMORY_VALUES:
            db.delete(memory)
            changed = True

    if changed:
        db.commit()


cleanup_db = SessionLocal()

try:
    cleanup_bad_memory_values(cleanup_db)
finally:
    cleanup_db.close()


# =========================================================
# MEMORY EXTRACTION
# =========================================================

def extract_memories(
    db,
    user_id,
    message,
):
    """
    IMPORTANT:

    Sirf declarative sentences memory mein save honge.

    Question:
        Mera naam kya hai?

    SAVE NAHI hoga.

    Statement:
        Mera naam Aahil hai.

    SAVE HOGA.
    """

    text = clean_text(message)

    if not text:
        return

    lower = text.lower()

    # -----------------------------------------------------
    # NAME
    # -----------------------------------------------------

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

            if name.lower() not in BAD_MEMORY_VALUES:

                save_memory(
                    db,
                    user_id,
                    "name",
                    name,
                )

                break

    # -----------------------------------------------------
    # LEARNING / WHAT USER IS LEARNING
    # -----------------------------------------------------

    learning_patterns = [
        r"\bmain\s+([a-zA-Z0-9+#.\-]+)\s+seekh\s+raha\s+hoon\b",
        r"\bmain\s+([a-zA-Z0-9+#.\-]+)\s+seekh\s+raha\s+hun\b",
        r"\bmain\s+([a-zA-Z0-9+#.\-]+)\s+seekh\s+raha\b",
        r"\bi\s+am\s+learning\s+([a-zA-Z0-9+#.\-]+)\b",
        r"\bi\s+am\s+learning\s+([a-zA-Z0-9+#.\- ]+)",
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
            )

            learning = learning.rstrip(
                ".,!? "
            )

            if learning.lower() not in BAD_MEMORY_VALUES:

                save_memory(
                    db,
                    user_id,
                    "learning",
                    learning,
                )

                break


# =========================================================
# MEMORY QUESTION DETECTION
# =========================================================

def is_name_question(text):

    lower = text.lower().strip()

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

    lower = text.lower().strip()

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


def is_combined_memory_question(text):

    return (
        is_name_question(text)
        and is_learning_question(text)
    )


# =========================================================
# MEMORY ANSWERS
# =========================================================

def answer_memory_question(
    db,
    user_id,
    message,
):

    if is_combined_memory_question(message):

        name = get_memory_value(
            db,
            user_id,
            "name",
        )

        learning = get_memory_value(
            db,
            user_id,
            "learning",
        )

        if name and learning:

            return (
                f"Tumhara naam {name} hai. "
                f"Tum {learning} seekh rahe ho."
            )

        if name:

            return (
                f"Tumhara naam {name} hai, "
                f"lekin tum kya seekh rahe ho "
                f"ye mujhe abhi yaad nahi."
            )

        if learning:

            return (
                f"Tum {learning} seekh rahe ho, "
                f"lekin tumhara naam mujhe abhi yaad nahi."
            )

        return (
            "Mujhe abhi tumhara naam ya tum kya "
            "seekh rahe ho, dono yaad nahi."
        )

    if is_name_question(message):

        name = get_memory_value(
            db,
            user_id,
            "name",
        )

        if name:

            return f"Tumhara naam {name} hai."

        return (
            "Tumne abhi mujhe apna naam nahi bataya."
        )

    if is_learning_question(message):

        learning = get_memory_value(
            db,
            user_id,
            "learning",
        )

        if learning:

            return (
                f"Tum {learning} seekh rahe ho."
            )

        return (
            "Tum kya seekh rahe ho, ye mujhe abhi yaad nahi."
        )

    return None


# =========================================================
# PROPERTY HELPERS
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
            Property.id == active.property_id
        )
        .first()
    )


def set_active_property(
    db,
    conversation_id,
    property_id,
):

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
        active.updated_at = datetime.utcnow()

    else:

        active = ActiveProperty(
            conversation_id=conversation_id,
            property_id=property_id,
        )

        db.add(active)

    db.commit()


def set_comparison(
    db,
    conversation_id,
    properties,
):

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
        comparison.updated_at = datetime.utcnow()

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

    for item in comparison.property_ids.split(","):

        item = item.strip()

        if item.isdigit():
            ids.append(int(item))

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
        property_obj.id: property_obj
        for property_obj in properties
    }

    return [
        property_map[property_id]
        for property_id in ids
        if property_id in property_map
    ]


# =========================================================
# PROPERTY SEARCH
# =========================================================

def extract_number(text):

    match = re.search(
        r"\b(\d+(?:\.\d+)?)\b",
        text,
    )

    if not match:
        return None

    try:
        return float(match.group(1))
    except Exception:
        return None


def search_properties(
    db,
    message,
):

    lower = message.lower()

    query = (
        db.query(Property)
        .filter(
            Property.status == "available"
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
        "lahore",
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
    # SIZE
    # -----------------------------------------------------

    size_match = re.search(
        r"(\d+)\s*marla",
        lower,
    )

    if size_match:

        size = int(
            size_match.group(1)
        )

        query = query.filter(
            Property.size_marla == size
        )

    # -----------------------------------------------------
    # PROPERTY TYPE
    # -----------------------------------------------------

    if "house" in lower:

        query = query.filter(
            Property.property_type.ilike(
                "%house%"
            )
        )

    elif "plot" in lower:

        query = query.filter(
            Property.property_type.ilike(
                "%plot%"
            )
        )

    elif "flat" in lower or "apartment" in lower:

        query = query.filter(
            Property.property_type.ilike(
                "%flat%"
            )
        )

    # -----------------------------------------------------
    # PURPOSE
    # -----------------------------------------------------

    if (
        "rent" in lower
        or "rental" in lower
        or "kiraye" in lower
    ):

        query = query.filter(
            Property.purpose.ilike("%rent%")
        )

    elif (
        "sale" in lower
        or "buy" in lower
        or "purchase" in lower
        or "chahiye" in lower
    ):

        query = query.filter(
            Property.purpose.ilike("%sale%")
        )

    return query.order_by(
        Property.price.asc()
    ).all()


# =========================================================
# PROPERTY LIST RESPONSE
# =========================================================

def format_property_list(
    properties,
):

    lines = [
        f"Mujhe {len(properties)} matching properties mili hain:"
    ]

    for index, property_obj in enumerate(
        properties,
        start=1,
    ):

        lines.append(
            f"\n{index}. {property_obj.title}"
        )

        lines.append(
            f"   Price: {money_lakh(property_obj.price)} lakh"
        )

        lines.append(
            f"   Size: {property_obj.size_marla} marla"
        )

        lines.append(
            f"   Bedrooms: {property_obj.bedrooms}"
        )

        lines.append(
            f"   Bathrooms: {property_obj.bathrooms}"
        )

        lines.append(
            f"   Location: {property_obj.location}"
        )

    return "\n".join(lines)


# =========================================================
# PROPERTY DETAIL
# =========================================================

def format_property_details(
    property_obj,
):

    return (
        f"{property_obj.title}\n\n"
        f"Property type: {property_obj.property_type}\n"
        f"Purpose: {property_obj.purpose}\n"
        f"Location: {property_obj.location}\n"
        f"Price: {money_lakh(property_obj.price)} lakh\n"
        f"Size: {property_obj.size_marla} marla\n"
        f"Bedrooms: {property_obj.bedrooms}\n"
        f"Bathrooms: {property_obj.bathrooms}\n"
        f"Description: {property_obj.description}\n"
        f"Status: {property_obj.status}\n"
        f"Owner: {property_obj.contact_name}\n"
        f"Contact: {property_obj.contact_phone}"
    )


# =========================================================
# PROPERTY QUESTION DETECTION
# =========================================================

def asks_property_search(text):

    lower = text.lower()

    property_words = [
        "house",
        "ghar",
        "property",
        "plot",
        "flat",
        "apartment",
    ]

    search_words = [
        "chahiye",
        "available",
        "find",
        "search",
        "looking",
        "dhoond",
        "dikhao",
        "show",
        "mil sakta",
        "mil sakti",
    ]

    return (
        any(
            word in lower
            for word in property_words
        )
        and any(
            word in lower
            for word in search_words
        )
    )


def asks_compare(text):

    lower = text.lower()

    return (
        "compare" in lower
        or "comparison" in lower
        or "dono compare" in lower
        or "dono ko compare" in lower
        or "muqabla" in lower
        or "mukabla" in lower
    )


def asks_cheaper(text):

    lower = text.lower()

    return (
        "sasti kaunsi" in lower
        or "sasta kaunsa" in lower
        or "cheaper" in lower
        or "cheapest" in lower
        or "kam price" in lower
        or "kam qeemat" in lower
    )


def asks_more_bedrooms(text):

    lower = text.lower()

    return (
        "zyada bedrooms" in lower
        or "zyaada bedrooms" in lower
        or "more bedrooms" in lower
        or "kis mein zyada bedroom" in lower
        or "kis me zyada bedroom" in lower
    )


def asks_owner_phone(text):

    lower = text.lower()

    return (
        "owner ka number" in lower
        or "owner ka phone" in lower
        or "owner number" in lower
        or "contact number" in lower
        or "contact do" in lower
        or "number do" in lower
    )


def asks_price(text):

    lower = text.lower()

    return (
        "price kya hai" in lower
        or "price batao" in lower
        or "kitne ki hai" in lower
        or "kitnay ki hai" in lower
        or "kitne ka hai" in lower
        or "kitnay ka hai" in lower
        or "cost kya hai" in lower
    )


def asks_details(text):

    lower = text.lower()

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

    lower = message.lower()

    if (
        "pehli" in lower
        or "pahli" in lower
        or "first" in lower
        or "1st" in lower
    ):

        if len(comparison) >= 1:
            return comparison[0]

    if (
        "doosri" in lower
        or "dusri" in lower
        or "second" in lower
        or "2nd" in lower
    ):

        if len(comparison) >= 2:
            return comparison[1]

    if (
        "teesri" in lower
        or "third" in lower
        or "3rd" in lower
    ):

        if len(comparison) >= 3:
            return comparison[2]

    return None


# =========================================================
# COMPARISON RESPONSE
# =========================================================

def build_comparison(
    properties,
):

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
            f"{index}. {property_obj.title}"
        )

        lines.append(
            f"   Price: {money_lakh(property_obj.price)} lakh"
        )

        lines.append(
            f"   Size: {property_obj.size_marla} marla"
        )

        lines.append(
            f"   Bedrooms: {property_obj.bedrooms}"
        )

        lines.append(
            f"   Bathrooms: {property_obj.bathrooms}"
        )

        lines.append(
            f"   Location: {property_obj.location}"
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
        expensive.price - cheapest.price
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
        if item.bedrooms == max_bedrooms
    ]

    if len(bedroom_winners) == len(properties):

        lines.append(
            "Dono mein bedrooms ki tadaad same hai."
        )

    else:

        lines.append(
            "Zyada bedrooms: "
            + ", ".join(
                item.title
                for item in bedroom_winners
            )
        )

    return "\n".join(lines)


# =========================================================
# PROPERTY CONTEXT QUESTIONS
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

    lower = message.lower()

    # -----------------------------------------------------
    # OWNER PHONE
    # -----------------------------------------------------

    if asks_owner_phone(message):

        return (
            f"{active_property.title} ke "
            f"{active_property.contact_name} ka "
            f"contact number: "
            f"{active_property.contact_phone}"
        )

    # -----------------------------------------------------
    # PRICE
    # -----------------------------------------------------

    if asks_price(message):

        return (
            f"{active_property.title} ki price "
            f"{money_lakh(active_property.price)} lakh hai."
        )

    # -----------------------------------------------------
    # DETAILS
    # -----------------------------------------------------

    if asks_details(message):

        return format_property_details(
            active_property
        )

    # -----------------------------------------------------
    # BEDROOMS
    # -----------------------------------------------------

    if (
        "bedrooms" in lower
        or "bedroom" in lower
    ):

        return (
            f"{active_property.title} mein "
            f"{active_property.bedrooms} bedrooms hain."
        )

    # -----------------------------------------------------
    # BATHROOMS
    # -----------------------------------------------------

    if (
        "bathrooms" in lower
        or "bathroom" in lower
    ):

        return (
            f"{active_property.title} mein "
            f"{active_property.bathrooms} bathrooms hain."
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
            Memory.user_id == user_id
        )
        .order_by(
            Memory.timestamp.desc()
        )
        .all()
    )

    memory_lines = []

    for memory in memories:

        value = clean_text(
            memory.value
        )

        if value.lower() in BAD_MEMORY_VALUES:
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
        .limit(10)
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

    property_context = "No active property."

    if active_property:

        property_context = (
            f"Title: {active_property.title}\n"
            f"Price: {money_lakh(active_property.price)} lakh\n"
            f"Size: {active_property.size_marla} marla\n"
            f"Bedrooms: {active_property.bedrooms}\n"
            f"Bathrooms: {active_property.bathrooms}\n"
            f"Location: {active_property.location}\n"
            f"Owner: {active_property.contact_name}\n"
            f"Phone: {active_property.contact_phone}\n"
            f"Description: {active_property.description}"
        )

    system_prompt = """
You are an intelligent property AI assistant.

The user can speak English, Urdu, or Roman Urdu.

Rules:

1. Never invent user memory.
2. Never invent property information.
3. Never change saved facts.
4. If exact information exists in the provided context, use it.
5. Keep answers concise and natural.
6. Do not say that you are an AI language model unless necessary.
7. For property questions, use the property context.
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
            data.get("message", {})
            .get("content", "")
            .strip()
        )

        if reply:
            return reply

        return (
            "Mujhe response generate karne mein "
            "problem hui."
        )

    except requests.exceptions.ConnectionError:

        return (
            "Ollama se connection nahi ho raha. "
            "Check karo ke Ollama running hai aur "
            f"{OLLAMA_MODEL} model installed hai."
        )

    except requests.exceptions.Timeout:

        return (
            "Ollama response dene mein zyada time le raha hai. "
            "Dobara try karo."
        )

    except Exception as exc:

        print(
            "OLLAMA ERROR:",
            repr(exc),
        )

        return (
            "AI service se temporary problem aa gayi. "
            "Please dobara try karo."
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

    text = clean_text(message)

    lower = text.lower()

    # =====================================================
    # 1. MEMORY EXTRACTION
    # =====================================================

    extract_memories(
        db,
        user_id,
        text,
    )

    # =====================================================
    # 2. MEMORY QUESTIONS
    # =====================================================

    memory_reply = answer_memory_question(
        db,
        user_id,
        text,
    )

    if memory_reply:

        return memory_reply

    # =====================================================
    # 3. PROPERTY COMPARISON QUESTIONS
    # =====================================================

    comparison_properties = (
        get_comparison_properties(
            db,
            conversation_id,
        )
    )

    if asks_compare(text):

        if len(comparison_properties) >= 2:

            return build_comparison(
                comparison_properties
            )

        # Try active search context

        active = get_active_property(
            db,
            conversation_id,
        )

        if active:

            return (
                "Comparison ke liye 2 properties "
                "chahiye. Pehle 2 properties search "
                "karo."
            )

        return (
            "Comparison ke liye pehle kam az kam "
            "2 properties search karo."
        )

    if asks_cheaper(text):

        if len(comparison_properties) >= 2:

            cheapest = min(
                comparison_properties,
                key=lambda item: item.price,
            )

            return (
                f"{cheapest.title} sasti hai. "
                f"Iski price "
                f"{money_lakh(cheapest.price)} lakh hai."
            )

        return (
            "Abhi comparison ke liye 2 properties "
            "available nahi hain."
        )

    if asks_more_bedrooms(text):

        if len(comparison_properties) >= 2:

            max_bedrooms = max(
                item.bedrooms
                for item in comparison_properties
            )

            winners = [
                item
                for item in comparison_properties
                if item.bedrooms == max_bedrooms
            ]

            if len(winners) == len(
                comparison_properties
            ):

                return (
                    f"Dono properties mein "
                    f"{max_bedrooms} bedrooms hain, "
                    f"isliye bedrooms ki tadaad same hai."
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
            "Pehle kam az kam 2 properties search "
            "karo, phir main bedrooms compare kar dunga."
        )

    # =====================================================
    # 4. NUMBERED PROPERTY SELECTION
    # =====================================================

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
                f"{money_lakh(selected.price)} lakh hai."
            )

        if asks_details(text):

            return format_property_details(
                selected
            )

        return format_property_details(
            selected
        )

    # =====================================================
    # 5. ACTIVE PROPERTY QUESTIONS
    # =====================================================

    active_reply = answer_active_property_question(
        db,
        conversation_id,
        text,
    )

    if active_reply:

        return active_reply

    # =====================================================
    # 6. PROPERTY SEARCH
    # =====================================================

    if asks_property_search(text):

        properties = search_properties(
            db,
            text,
        )

        if not properties:

            return (
                "Is waqt mujhe tumhari requirements "
                "ke mutabiq koi available property nahi mili."
            )

        # Save first property as active context

        set_active_property(
            db,
            conversation_id,
            properties[0].id,
        )

        # Save all search results for comparison

        set_comparison(
            db,
            conversation_id,
            properties,
        )

        return format_property_list(
            properties
        )

    # =====================================================
    # 7. GENERAL OLLAMA CHAT
    # =====================================================

    return ollama_reply(
        db,
        user_id,
        conversation_id,
        text,
    )


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

        return {
            "message": (
                "Enterprise Property AI Chatbot "
                "is running!"
            ),
            "version": APP_VERSION,
            "ai_model": OLLAMA_MODEL,
            "ai_provider": "Ollama",
            "conversation_context": True,
            "context_messages": 10,
            "memory": True,
            "property_system": True,
            "active_property_context": True,
            "property_comparison": True,
            "database": "SQLite",
            "properties": property_count,
        }

    finally:

        db.close()


# =========================================================
# CREATE USER
# =========================================================

@app.post("/users")
def create_user(name: str):

    db = SessionLocal()

    try:

        name = clean_text(name)

        if not name:

            raise HTTPException(
                status_code=400,
                detail="Name cannot be empty",
            )

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


# =========================================================
# GET USERS
# =========================================================

@app.get("/users")
def get_users():

    db = SessionLocal()

    try:

        users = (
            db.query(User)
            .order_by(User.id)
            .all()
        )

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


# =========================================================
# CREATE CONVERSATION
# =========================================================

@app.post("/conversations/{user_id}")
def create_conversation(
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

        conversation = Conversation(
            user_id=user_id
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
                detail="Conversation not found",
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
                "Chat processing error. "
                f"{str(exc)}"
            ),
        )

    finally:

        db.close()


# =========================================================
# PROPERTIES - CREATE
# =========================================================

@app.post("/properties")
def create_property(
    request: PropertyCreate,
):

    db = SessionLocal()

    try:

        property_obj = Property(
            title=clean_text(
                request.title
            ),
            property_type=clean_text(
                request.property_type
            ),
            purpose=clean_text(
                request.purpose
            ),
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
            ),
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
# PROPERTIES - GET
# =========================================================

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
                property_to_dict(
                    property_obj
                )
                for property_obj in properties
            ],
        }

    finally:

        db.close()


# =========================================================
# PROPERTY BY ID
# =========================================================

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


# =========================================================
# USER MEMORY
# =========================================================

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
                Memory.user_id == user_id
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


# =========================================================
# CONVERSATION MESSAGES
# =========================================================

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
                Conversation.id
                == conversation_id
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
                == conversation_id
            )
            .order_by(Message.id)
            .all()
        )

        return {
            "conversation_id": conversation_id,
            "messages": [
                {
                    "id": item.id,
                    "user_message": item.user_message,
                    "bot_reply": item.bot_reply,
                    "timestamp": item.timestamp,
                }
                for item in messages
            ],
        }

    finally:

        db.close()


# =========================================================
# ACTIVE PROPERTY
# =========================================================

@app.get(
    "/conversations/{conversation_id}/active-property"
)
def get_active_property_api(
    conversation_id: int,
):

    db = SessionLocal()

    try:

        property_obj = get_active_property(
            db,
            conversation_id,
        )

        return {
            "conversation_id": conversation_id,
            "active_property": (
                property_to_dict(property_obj)
                if property_obj
                else None
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
):

    db = SessionLocal()

    try:

        properties = get_comparison_properties(
            db,
            conversation_id,
        )

        return {
            "conversation_id": conversation_id,
            "count": len(properties),
            "comparison": (
                build_comparison(properties)
                if len(properties) >= 2
                else "Comparison ke liye 2 properties nahi hain."
            ),
            "properties": [
                property_to_dict(
                    property_obj
                )
                for property_obj in properties
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
        port=8000,
    )