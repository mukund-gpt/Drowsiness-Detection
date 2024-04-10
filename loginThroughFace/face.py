import numpy as np
import cv2
import face_recognition
import sqlite3

#database
conn=sqlite3.connect('db.sqlite3')
cursor=conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS user
                (id INTEGER PRIMARY KEY, name TEXT,face_encoding BLOB)''')

# Check if face encoding exists in the database
def face_encoding_exists(face_encoding):
    cursor.execute('''SELECT face_encoding FROM user''')
    existing_encodings = cursor.fetchall()
    for row in existing_encodings:
        existing_encoding_bytes = row[0]
        if existing_encoding_bytes == face_encoding:
            return True
    return False


# Insert person data
def insert_person_data(name, face_encoding):
    face_encoding_bytes = face_encoding.tobytes()  # Convert face encoding to bytes
    if not face_encoding_exists(face_encoding_bytes):
        cursor.execute('''INSERT INTO user (name, face_encoding) VALUES (?, ?)''', (name, face_encoding_bytes))
        conn.commit()
        print(f"Person '{name}' added to the database.")
    else:
        print("Person with similar face data already exists in the database.")



# Load images
img1 = face_recognition.load_image_file("elon1.jpeg")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img2 = face_recognition.load_image_file("elon2.jpeg")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# Detect face locations and encodings
face_loc1 = face_recognition.face_locations(img1)[0]
encode1 = face_recognition.face_encodings(img1)[0]
cv2.rectangle(img1, (face_loc1[3], face_loc1[0]), (face_loc1[1], face_loc1[2]), (255, 0, 255), 2)

face_loc2 = face_recognition.face_locations(img2)[0]
encode2 = face_recognition.face_encodings(img2)[0]
cv2.rectangle(img2, (face_loc2[3], face_loc2[0]), (face_loc2[1], face_loc2[2]), (255, 0, 255), 2)

# Compare face encodings
results = face_recognition.compare_faces([encode1], encode2)
print(results)

# Display images
# cv2.imshow("elon1", img_elon1)
# cv2.imshow("elon2", img_elon2)
cv2.waitKey(0)


insert_person_data("Elon Musk", encode1)
insert_person_data("Elon Musk", encode2)

# cursor.execute("SELECT * FROM user")
# print(cursor.fetchall())
conn.close()