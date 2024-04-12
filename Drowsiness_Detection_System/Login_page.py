from time import sleep
import time
import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import Tk, Label, Button
from imutils import face_utils
import dlib
from pygame import mixer
import numpy as np
import hand_tracking_module as htm
import os
import face_recognition
from PIL import ImageGrab


mixer.init()

cap = None
root = tk.Tk()
root.geometry("1200x700")
root.title("Login Page")



def update_frame():
          
          global cap
          success, frame = cap.read()
          
          rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          pil_image = Image.fromarray(rgb_image)
          tk_image = ImageTk.PhotoImage(pil_image)
          imgFrame.config(image=tk_image)
          imgFrame.image = tk_image

          imgFrame.after(10, update_frame)  # Update frame every 10 milliseconds
        
                 
def videoStart():
    global cap
    if cap:
        cap.release()
    
    cap = cv2.VideoCapture(0)
    cap.set(3, 1200)
    cap.set(4, 700)
    update_frame()           
    
def goToGUIPage():
    screenshot = ImageGrab.grab()
    screenshot_np = np.array(screenshot)
    folder_path = "User_Images"
    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)
        imgAlreadyExisting = face_recognition.load_image_file(file_path)
        # cv2.imshow("Already Existing", imgAlreadyExisting)
        face_loc1 = face_recognition.face_locations(imgAlreadyExisting)[0]
        imgAlreadyExistingEnc = face_recognition.face_encodings(imgAlreadyExisting)[0]
        # cv2.imshow("Screenshot image", screenshot_np)
        face_loc2 = face_recognition.face_locations(screenshot_np)[0]
        imgNewEnc = face_recognition.face_encodings(screenshot_np)[0]
        results = face_recognition.compare_faces([imgAlreadyExistingEnc], imgNewEnc)
        print(results)
        if results[0]:
            mixer.music.load("Already_registered.wav")
            mixer.music.play()
            sleep(1)
            print("Already Registered")
            
            import gui1
            root.destroy()
            break
        else:
            mixer.music.load("register.wav")
            mixer.music.play()
            print("Not registered")
            if cap:
                print("Cap exists1")
            else:
                print("It doesn't exists1")
            cap.release()
            if cap:
                print("Cap exists2")
            else:
                print("It doesn't exists2")
            cv2.destroyAllWindows()
            if cap:
                print("Cap exists3")
            else:
                print("It doesn't exists3")
            # import gui1
            import register
            root.destroy()
            
                    
def goToRegisterPage():
    import register               

imgFrame = Label(root, height=700, width=1200, bg="blue")
imgFrame.place(x=0, y=0)           

loginButton = Button(root, text="Login", width=20, height=2, command=goToGUIPage)
loginButton.place(x=1100, y=640)

registerButton = Button(root, text="Register", width=20, height=2, command=goToRegisterPage)
registerButton.place(x=10, y=640)

videoStart()         

# playButton = Button(root, text="Play", width=20, height=2, command=videoStart)
# playButton.place(x=10, y=640)



root.mainloop()