import jetson.inference
import jetson.utils
import time
import numpy as np
import cv2
import pyttsx3

timeStamp=time.time()
fpsFiltered=0

net = jetson.inference.detectNet('ssd-mobilenet-v2',threshold=0.5)
dispW=1280
dispH=720
font=cv2.FONT_HERSHEY_SIMPLEX
ConfidenceThreshold = 0.85
flip=2
# piCamSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! videobalance  hue=0.1 contrast=1.5 brightness=-.3 saturation=1.2 ! appsink  drop=true'
piCamSet  = 'nvarguscamerasrc'
piCamSet += ' ! video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1'
piCamSet += ' ! nvvidconv flip-method='+str(flip)
piCamSet += ' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx'
piCamSet += ' ! videoconvert'
piCamSet += ' ! video/x-raw, format=BGR'
piCamSet += ' ! videobalance  hue=0.1 contrast=1.5 brightness=-.3 saturation=1.2'
piCamSet += ' ! appsink  drop=true'
picam = cv2.VideoCapture(piCamSet)

webcam1 = cv2.VideoCapture('/dev/video1')
webcam1.set(cv2.CAP_PROP_FRAME_WIDTH,dispW)
webcam1.set(cv2.CAP_PROP_FRAME_HEIGHT,dispH)

webcam2 = cv2.VideoCapture('/dev/video2')
webcam2.set(cv2.CAP_PROP_FRAME_WIDTH,dispW)
webcam2.set(cv2.CAP_PROP_FRAME_HEIGHT,dispH)

speechEngine = pyttsx3.init()
speechEngine.setProperty('rate',150)

cam = picam
#cam = webcam1
#cam = webcam2
identified = []
while True:
    _,frame=cam.read()
    width =frame.shape[0]
    height=frame.shape[1]
    img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA).astype(np.float32)
    img=jetson.utils.cudaFromNumpy(img)
    detections = net.Detect(img,width,height)
    for detection in detections:
        ClassId=detection.ClassID
        Confidence=detection.Confidence
        print('Confidence, Threshold',Confidence,ConfidenceThreshold)
        if Confidence > ConfidenceThreshold:
            item=net.GetClassDesc(ClassId)

            top    = int(detection.Top)
            bottom = int(detection.Bottom)
            left   = int(detection.Left)
            right  = int(detection.Right)
            cv2.rectangle(frame,(left,top),(right,bottom),(0,255,0),1)
            cv2.putText(frame,item,(left,top+20),font,1,(0,0,255),2)
            if not item in identified:
                identified.append(item)
                speechEngine.say(item)
                speechEngine.runAndWait()
    dt = time.time() - timeStamp
    timeStamp=time.time()
    fps = 1/dt
    fpsFiltered = 0.9*fpsFiltered + 0.1*fps
    fpsMessage='{} fps'.format(round(fpsFiltered,1))
    cv2.putText(frame,fpsMessage,(0,30),font,1,(0,0,255),2)
    cv2.imshow('recCam',frame)
    cv2.moveWindow('recCam',0,0)
    if cv2.waitKey(1)==ord('q'):
        break
picam.release()
webcam1.release()
webcam2.release()
cv2.destroyAllWindows()
