import sqlite3

# Connect to your database
conn = sqlite3.connect('chatbot.db')
cursor = conn.cursor()

# Update user with id 15 to admin role
cursor.execute("UPDATE users SET role = 'admin' WHERE id = 15;")
conn.commit()

# Verify the update
cursor.execute("SELECT id, name, email, role FROM users WHERE id = 15;")
user = cursor.fetchone()
print("Updated User:", user)

conn.close()