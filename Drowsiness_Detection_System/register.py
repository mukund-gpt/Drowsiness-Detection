import hand_tracking_module as htm
import mediapipe
import cv2
import math
from time import sleep
import numpy as np
import cvzone
from pygame import mixer
from time import sleep
import time
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import Tk, Label, Button
import os
import sqlite3

# Connect to SQLite database (creates a new database if it doesn't exist)
conn = sqlite3.connect('db.sqlite3')
# Create a cursor object
cursor = conn.cursor()
# Create a table
cursor.execute('''CREATE TABLE IF NOT EXISTS user
                (id INTEGER PRIMARY KEY, 
                username TEXT DEFAULT 'PRAKHAR'
                )''')

mixer.init()

cap = None
img = None
finalText = ""
keySound = mixer.Sound("key_pressed.wav")
root = tk.Tk()
root.geometry("1280x900")
root.title("Register Page")

keys = None
detector = htm.handDetector()

keys = [[["!"], ["@"], ["#"], ["$"], ["%"], ["^"], ["&"], ["*"], ["("], [")"]],
        [["Q"], ["W"], ["E"], ["R"], ["T"], ["Y"], ["U"], ["I"], ["O"], ["P"]],
        [["A"], ["S"], ["D"], ["F"], ["G"], ["H"], ["J"], ["K"], ["L"], [":"]],
        [["Z"], ["X"], ["C"], ["V"], ["B"], ["N"], ["M"], ["<"], [">"], ["?"]]
        ]

buttonList = []

def findDistance(x1, x2, y1, y2):
          dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
          return dist

def drawAll(img, buttonList):
        imgNew = np.zeros_like(img, np.uint8)
        for btn in buttonList:
                x, y = btn.pos
                w, h = btn.size
                cvzone.cornerRect(imgNew, (btn.pos[0], btn.pos[1], btn.size[0], btn.size[1]), l = 20, rt = 8)
                cv2.rectangle(imgNew, btn.pos, (x + w, y + h), (255, 0, 255), cv2.FILLED)
                if btn.text == "Space Bar":
                        cv2.putText(imgNew, "Space Bar", [x + 19, y + 73] , cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
                else:
                        cv2.putText(imgNew, str(btn.text[0]), [x + 19, y + 73] , cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
        
        out = img.copy()
        alpha = 0.5
        mask = imgNew.astype(bool)
        out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]
        return out

class Button1():
          def __init__(self, pos, text, size = [100, 100]):
                    global img
                    self.pos = pos
                    self.size = size
                    self.text = text

spaceBar = Button1([290, 490], "Space Bar", size = [640, 100])
# capsLock = Button([50, 490], "Caps Lock", size = [220, 100])

for i in range(0, 4):
                for j, key in enumerate(keys[i]):
                      buttonList.append(Button1([50 + 120 * j, 10 + 120 * i], key))
                buttonList.append(spaceBar)

def update_frame():
          
          global cap
          global img
          global finalText
          global buttonList
          success, img = cap.read()
          
          if success:
                img = cv2.flip(img, 1)
                img = detector.findHands(img)
                lmList = detector.findPosition(img, draw=False)

                img = drawAll(img, buttonList)

                if lmList:
                      for btn in buttonList:
                              x, y = btn.pos
                              w, h = btn.size
                              x8 = lmList[8][1]
                              y8 = lmList[8][2]
                              x12 = lmList[12][1]
                              y12 = lmList[12][2]
                              cx1, cy1 = (x8 + x12) // 2, (y8 + y12) // 2
                              img = cv2.circle(img, (x12, y12), 10, (255, 0, 255), cv2.FILLED)
                              img = cv2.circle(img, (x8, y8), 10, (255, 0, 255), cv2.FILLED)
                              img = cv2.circle(img, (cx1, cy1), 10, (255, 0, 255), cv2.FILLED)
                              img = cv2.line(img, (x8, y8), (x12, y12), (255, 0, 255), 3)
                              dist812 = findDistance(x8, x12, y8, y12)
                              if x < x8 < x + w and y < y8 < y + h:
                                      img = cv2.rectangle(img, [x, y], [x + w, y + h], (175, 0, 175), cv2.FILLED)
                                      img = cv2.putText(img, str(btn.text[0]), [x + 19, y + 73] , cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
                                      if dist812 < 30:
                                              keySound.play()
                                              # keyboard.press(btn.text)
                                              img = cv2.rectangle(img, [x, y], [x + w, y + h], (0, 255, 0), cv2.FILLED)
                                              img = cv2.putText(img, str(btn.text[0]), [x + 19, y + 73] , cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
                                              if btn.text != "Space Bar":
                                                      finalText += str((btn.text)[0])
                                              else:
                                                      finalText += " "
                                                      # if btn.text == "Space Bar":
                                                      #         finalText += " "
                                                      # else:
                                                      #         if caps:
                                                      #                 caps = False
                                                      #         else:
                                                      #                 caps = True
                                              sleep(1)

                img = cv2.rectangle(img, [200, 620], [950, 720], (255, 0, 255), cv2.FILLED)
                img = cv2.putText(img, finalText, [269, 670] , cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)  
                print(img)
                # rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(img)
                tk_image = ImageTk.PhotoImage(pil_image)
                imgFrame.config(image=tk_image)
                imgFrame.image = tk_image

          
          imgFrame.after(10, update_frame)  # Update frame every 10 milliseconds
          
def submitUsername():
        print(finalText)
        cursor.execute('''INSERT INTO user (username) VALUES (?)''', (finalText, ))
        conn.commit()
        import gui1
imgFrame = Label(root, height=900, width=1280, bg="lightgray")
imgFrame.place(x=0, y=0)    

submit1Button = Button(root, text="SUBMIT", width=20, height=2, command=submitUsername)
submit1Button.place(x=600, y=30)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 900)
update_frame()        

root.mainloop()