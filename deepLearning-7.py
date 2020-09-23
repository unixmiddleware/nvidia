import jetson.inference
import jetson.utils
import time
import numpy as np
import cv2

timeStamp=time.time()
fpsFiltered=0

onnx='/home/mark/Downloads/jetson-inference/python/training/classification/myModel/resnet18.onnx'
labf='/home/mark/Downloads/jetson-inference/myTrain/labels.txt'
net = jetson.inference.imageNet('alexnet',['--model='+onnx,'--input_blob=input_0','--output_blob=output_0','--labels='+labf])
dispW=1280
dispH=720
font=cv2.FONT_HERSHEY_SIMPLEX

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

#cam = picam
#cam = webcam1
cam = webcam2

while True:
    _,frame=cam.read()
    width =frame.shape[0]
    height=frame.shape[1]
    img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA).astype(np.float32)
    img=jetson.utils.cudaFromNumpy(img)
    classID,confidence = net.Classify(img,width,height)
    item=''
    item=net.GetClassDesc(classID)
    dt = time.time() - timeStamp
    timeStamp=time.time()
    fps = 1/dt
    fpsFiltered = 0.9*fpsFiltered + 0.1*fps
    fpsMessage='{} fps {}'.format(round(fpsFiltered,1),item)
    cv2.putText(frame,fpsMessage,(0,30),font,1,(0,0,255),2)
    cv2.imshow('recCam',frame)
    cv2.moveWindow('recCam',0,0)
    if cv2.waitKey(1)==ord('q'):
        break
picam.release()
webcam1.release()
webcam2.release()
cv2.destroyAllWindows()
