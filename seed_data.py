import os
from sqlalchemy import create_engine, text

# Aapki Railway Public Database URL
DATABASE_URL = "postgresql://postgres:PIPuAZryLEgVvsqyElUhKiAPYxcqzDdm@altaria.proxy.rlwy.net:53517/railway"

engine = create_engine(DATABASE_URL)

# 1. Sabse pehle table create karne ki query
create_table_query = text("""
CREATE TABLE IF NOT EXISTS properties (
    id SERIAL PRIMARY KEY,
    title VARCHAR,
    property_type VARCHAR,
    purpose VARCHAR,
    location VARCHAR,
    price FLOAT,
    price_lakh FLOAT,
    size_marla FLOAT,
    bedrooms INT,
    bathrooms INT,
    description TEXT,
    status VARCHAR,
    contact_name VARCHAR,
    contact_phone VARCHAR
);
""")

properties_data = [
    {
        "title": "5 Marla House for Sale in Lahore",
        "property_type": "house",
        "purpose": "sale",
        "location": "DHA Lahore",
        "price": 15000000,
        "price_lakh": 150,
        "size_marla": 5,
        "bedrooms": 3,
        "bathrooms": 3,
        "description": "Beautiful 5 marla house in DHA Lahore with modern construction, spacious rooms and parking.",
        "status": "available",
        "contact_name": "Aahil",
        "contact_phone": "03001234567"
    },
    {
        "title": "4 Bedroom House in DHA Lahore",
        "property_type": "house",
        "purpose": "sale",
        "location": "DHA Phase 6, Lahore",
        "price": 18000000,
        "price_lakh": 180,
        "size_marla": 10,
        "bedrooms": 4,
        "bathrooms": 4,
        "description": "Beautiful 4 bedroom family house in DHA Phase 6 Lahore. Modern construction, spacious rooms and excellent location.",
        "status": "available",
        "contact_name": "Aahil Real Estate",
        "contact_phone": "03001234567"
    },
    {
        "title": "Luxury 1 Kanal House in DHA Lahore",
        "property_type": "house",
        "purpose": "sale",
        "location": "DHA Phase 6, Lahore",
        "price": 35000000,
        "price_lakh": 350,
        "size_marla": 20,
        "bedrooms": 5,
        "bathrooms": 5,
        "description": "Luxury 1 kanal house in DHA Phase 6 Lahore with modern construction, spacious rooms, parking and premium finishing.",
        "status": "available",
        "contact_name": "Aahil Real Estate",
        "contact_phone": "03001234567"
    },
    {
        "title": "7 Marla House for Sale in DHA Lahore",
        "property_type": "house",
        "purpose": "sale",
        "location": "DHA Phase 5, Lahore",
        "price": 22000000,
        "price_lakh": 220,
        "size_marla": 7,
        "bedrooms": 4,
        "bathrooms": 4,
        "description": "Beautiful 7 marla modern house in DHA Phase 5 Lahore with parking and spacious rooms.",
        "status": "available",
        "contact_name": "Aahil Real Estate",
        "contact_phone": "03001234567"
    },
    {
        "title": "5 Marla House for Sale in DHA Lahore",
        "property_type": "house",
        "purpose": "sale",
        "location": "DHA Phase 5, Lahore",
        "price": 22000000,
        "price_lakh": 220,
        "size_marla": 5,
        "bedrooms": 4,
        "bathrooms": 4,
        "description": "Beautiful 5 marla modern house in DHA Phase 5 Lahore with parking and spacious rooms.",
        "status": "available",
        "contact_name": "Aahil Real Estate",
        "contact_phone": "03001234567"
    }
]

with engine.begin() as conn:
    # Table banayein
    conn.execute(create_table_query)
    print("✅ Table 'properties' created successfully (or already exists)!")

    # Data insert karein
    for p in properties_data:
        insert_query = text("""
            INSERT INTO properties (title, property_type, purpose, location, price, price_lakh, size_marla, bedrooms, bathrooms, description, status, contact_name, contact_phone)
            VALUES (:title, :property_type, :purpose, :location, :price, :price_lakh, :size_marla, :bedrooms, :bathrooms, :description, :status, :contact_name, :contact_phone)
        """)
        conn.execute(insert_query, p)

print("✅ All properties successfully inserted into Railway database!")