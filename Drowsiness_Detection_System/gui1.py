import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import Tk, Label, Button
from scipy.spatial import distance
from imutils import face_utils
import dlib
from pygame import mixer
import numpy as np
from datetime import datetime
import hand_tracking_module as htm
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import math
import time
import threading
from concurrent.futures import thread
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

mixer.init()
mixer.music.load("C:/Users/VICTUS/Documents/ML_PROJECTS/Open_CV_Projects/Hack36_dowsy/music.wav")

detector = htm.handDetector()

segmentor = SelfiSegmentation()

listImg = os.listdir("C:/Users/VICTUS/Documents/ML_PROJECTS/Open_CV_Projects/Hack36_dowsy/BG_images")
imgList = []
for imgPath in listImg:
          img = cv2.imread(f"C:/Users/VICTUS/Documents/ML_PROJECTS/Open_CV_Projects/Hack36_dowsy/BG_images/{imgPath}")
          imgList.append(img)
          
indexImg = 2

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

tipIds = [4, 8, 12, 16, 20]

def findDistance(x1, x2, y1, y2):
          dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
          return dist

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[14], mouth[18])
    C = distance.euclidean(mouth[12], mouth[16])
    mar = A / C
    return mar

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

thresh_eye = 0.25
thresh_mouth = 0.5
frame_check = 20
flag_eye = 0
flag_mouth = 0
myFlag = 0

volRange = volume.GetVolumeRange()
volume.SetMasterVolumeLevel(-20.0, None)
minVol = volRange[0]
maxVol = volRange[1]
volBar = 400
volPer = 0
vol= 0

colorR = (255, 0, 255)

volButtonCoord = [100, 450]

volPerCoord = [40, 450]
volBarCoord = [[50, 150], [85, 400]]

frame = None
changeVolByHandGestVar  = False

prevImageRectCoord = [[900, 50], [1000, 70]]
nextImageRectCoord = [[1050, 50], [1150, 70]]

dowsinessAlertCoord = [10, 150]

drowsyTypeAlertCoords = [[130, 250], [130, 350]]

timeCoord = [50, 90]

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("C:/Users/VICTUS/Documents/ML_PROJECTS/Open_CV_Projects/Hack36_dowsy/shape_predictor_68_face_landmarks.dat")

cap = None

root = tk.Tk()
root.geometry("1200x700")
root.title("OpenCV + Tkinter")
        
def controlVolume1():
    t1 = threading.Thread(target = controlVolume)        
    t1.start()
        
def controlVolume():
    global changeVolByHandGestVar
    pTime = time.time()
    cTime = 0
    
    changeVolByHandGestVar = True
    time.sleep(3)
    changeVolByHandGestVar = False     

def update_frame():
    global colorR
    
    global thresh_eye
    global thresh_mouth
    global frame_check
    global flag_eye
    global flag_mouth
    global myFlag
    
    global frame
    global lmList
    global changeVolByHandGestVar
    
    global volRange
    global minVol
    global maxVol
    global volBar
    global volPer
    global vol
        
    global volBarCoord    
    global volPerCoord
    global prevImageRectCoord
    global nextImageRectCoord
    
    global drowsyTypeAlertCoords
    global dowsinessAlertCoord
    global timeCoord
    
    global indexImg
    
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    imgBG = cv2.resize(imgList[indexImg], (frame.shape[1], frame.shape[0]))
    frame = segmentor.removeBG(frame, frame)
    
    frame = cv2.rectangle(frame, (prevImageRectCoord[0][0], prevImageRectCoord[0][1]), (prevImageRectCoord[1][0], prevImageRectCoord[1][1]), (0, 0, 255), cv2.FILLED)
    frame = cv2.rectangle(frame, (nextImageRectCoord[0][0], nextImageRectCoord[0][1]), (nextImageRectCoord[1][0], nextImageRectCoord[1][1]), (0, 0, 255), cv2.FILLED)
    
    if changeVolByHandGestVar:

        if len(lmList) != 0:
            x4, y4 = lmList[4][1], lmList[4][2]
            x8, y8 = lmList[8][1], lmList[8][2]
            cx, cy = (x4 + x8) // 2, (y4 + y8) // 2
            length = math.sqrt((x8 - x4)**2 + (y8 - y4)**2)
            vol = np.interp(length, [40, 290], [minVol, maxVol])
            volBar = np.interp(length, [40, 290], [400, 150])
            volPer = np.interp(length, [40, 290], [0, 100])
            volume.SetMasterVolumeLevel(vol, None)
            frame = cv2.circle(frame, (x4, y4), 15, (255, 0, 255), cv2.FILLED)
            frame = cv2.circle(frame, (x8, y8), 15, (255, 0, 255), cv2.FILLED)
            frame = cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            frame = cv2.line(frame, (x4, y4), (x8, y8), (255, 0, 255), 3)
            if length < 50:
                      frame = cv2.circle(frame, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
            frame = cv2.rectangle(frame, (volBarCoord[0][0], volBarCoord[0][1]), (volBarCoord[1][0], volBarCoord[1][1]), (0, 255, 0), 3)
            frame = cv2.rectangle(frame, (volBarCoord[0][0], int(volBar)), (volBarCoord[1][0], volBarCoord[1][1]), (0, 255, 0), cv2.FILLED)
            frame = cv2.putText(frame, f'{int(volPer)} %', (volPerCoord[0], volPerCoord[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
    
    # if len(lmList) != 0:
    #     x12, y12 = lmList[12][1], lmList[12][2]
    #     cx1, cy1 = (x12 + x8) // 2, (y12 + y8) // 2
        
    #     img = cv2.circle(img, (x12, y12), 25, (255, 0, 255), cv2.FILLED)
    #     img = cv2.circle(img, (x8, y8), 25, (255, 0, 255), cv2.FILLED)
    #     img = cv2.circle(img, (cx1, cy1), 35, (255, 0, 255), cv2.FILLED)
    #     img = cv2.line(img, (x8, y8), (x12, y12), (255, 0, 255), 3)
        
    #     dist812 = findDistance(x8, x12, y8, y12)
        
    #     dist812 = findDistance(x8, x12, y8, y12)
    #     cursor = lmList[8]
    #     if dist812 < 45:
    #               if (cx - w // 2 < cursor[1] < cx + w // 2) and (cy - h // 2 < cursor[2] < cy + h // 2):
    #                         colorR  = (0, 255, 0)
    #                         cx = cursor[1]
    #                         cy = cursor[2]
    #     else:
    #               colorR = (255, 0, 255)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        cv2.putText(frame, "****************NO FACE DETECTED!****************", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")  
    
    cv2.putText(frame, current_time, (timeCoord[0], timeCoord[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
    
    height, width = frame.shape[:2]
    size = (width, height)
    
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw = False)
    

    for subject in subjects:
        
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]
        
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        
        mar = mouth_aspect_ratio(mouth)
        
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        
        if ear < thresh_eye:
            flag_eye += 1
            if flag_eye >= frame_check:
                cv2.putText(frame, "****************DROWSINESS ALERT!****************", (dowsinessAlertCoord[0], dowsinessAlertCoord[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 5)
                
                cv2.putText(frame, "****************EYES CLOSED ALERT!****************", (drowsyTypeAlertCoords[0][0], drowsyTypeAlertCoords[0][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.putText(frame, "****************EYES CLOSED ALERT!****************", (drowsyTypeAlertCoords[1][0], drowsyTypeAlertCoords[1][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                mixer.music.play()
        else:
            flag_eye = 0
        
        if mar > thresh_mouth:
            flag_mouth += 1
            if flag_mouth >= frame_check:
                cv2.putText(frame, "****************DROWSINESS ALERT!****************", (dowsinessAlertCoord[0], dowsinessAlertCoord[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 5)
                cv2.putText(frame, "****************YAWNING ALERT!****************", (drowsyTypeAlertCoords[0][0], drowsyTypeAlertCoords[0][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.putText(frame, "****************YAWNING ALERT!****************", (drowsyTypeAlertCoords[1][0], drowsyTypeAlertCoords[1][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                mixer.music.play()
                
                
                # fingersOpen = 0
                # print(5)
                # while True:
                #     mixer.music.play()
                #     if len(lmList) != 0:
                #         for id in range(0, 5):
                #              if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                #                 fingersOpen += 1
                #         if fingersOpen == 3:
                #             break
                #         else:
                #             fingersOpen = 0
                                        
                
        else:
            flag_mouth = 0
    
    
    
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    pil_image = Image.fromarray(rgb_image)
    tk_image = ImageTk.PhotoImage(pil_image)
    imgFrame.config(image=tk_image)
    imgFrame.image = tk_image 
        
    imgFrame.after(10, update_frame)

def videoStart():
    global cap
    if cap:
        cap.release()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 630)
    update_frame()  

def videoPaused():
          print("Paused Button Clicked")
          cap.release()   
          cv2.destroyAllWindows() 

# Tkinter GUI elements
imgFrame = Label(root, height=630, width=1200, bg="lightblue")
imgFrame.place(x=0, y=0)

playButton = Button(root, text="Play", width=20, height=2, command=videoStart)
playButton.place(x=10, y=640)

controlVolumeButton = Button(root, text="Volume", width=20, height=2, command=controlVolume1)
controlVolumeButton.place(x=400, y=640)

pauseButton = Button(root, text="Stop", width=20, height=2, command=videoPaused)
pauseButton.place(x=1000, y=640)

# Run the tkinter main loop
root.mainloop()
