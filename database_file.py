import sqlite3


# Connect to SQLite database (creates a new database if it doesn't exist)
conn = sqlite3.connect('db.sqlite3')
# Create a cursor object
cursor = conn.cursor()
# Create a table
cursor.execute('''CREATE TABLE IF NOT EXISTS user
                (id INTEGER PRIMARY KEY, name TEXT,face_encoding BLOB)''')
# Insert a row of data
cursor.execute("INSERT INTO user VALUES (1, 'Hello world','rhhberfhe')")
# Save (commit) the changes
conn.commit()
# Query the database
cursor.execute("SELECT * FROM user")
print(cursor.fetchall())
# Close the connection
conn.close()