import jetson.inference
import jetson.utils
import time
import cv2
import numpy as np

width = 1280
height = 720
width = 640
height = 360

#webcam1=jetson.utils.gstCamera(width,height,'/dev/video1') # 2.7 fps
#webcam2=jetson.utils.gstCamera(width,height,'/dev/video2') # 6.1 fps
#picam=jetson.utils.gstCamera(width,height,'0')             # 29.2 fps

dispW=width
dispH=height
flip=2
framerate=21
camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate='+str(framerate) +'/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
picam = cv2.VideoCapture(camSet)
webcam1 = cv2.VideoCapture('/dev/video1')
webcam2 = cv2.VideoCapture('/dev/video2')
net = jetson.inference.imageNet('googlenet')
font = cv2.FONT_HERSHEY_SIMPLEX
StartTime = time.time()
fpsFilter = 0
while True:
#    _,frame = webcam1.read()
#    _,frame = webcam2.read()
    _,frame = picam.read()
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA).astype(np.float32)
    img = jetson.utils.cudaFromNumpy(img)
    classID,confidence = net.Classify(img,width,height)
    item=net.GetClassDesc(classID)
    dt = time.time() - StartTime
    fps = 1/dt
    fpsFilter = 0.95*fpsFilter + 0.05*fps
    StartTime = time.time()
    title = str(round(fpsFilter,1)) + ' fps ' + item
    cv2.putText(frame,title,(0,30),font,1,(0,0,255),2)
    cv2.imshow('win',frame)
    cv2.moveWindow('win',0,0)
    
    if cv2.waitKey(1) == ord('q'):
        break

webcam1.release()
webcam2.release()
picam.release()
cv2.destroyAllWindows()
