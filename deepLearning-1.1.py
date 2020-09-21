import jetson.inference
import jetson.utils
import time
import cv2
import numpy as np

width = 1280
height = 720

webcam1=jetson.utils.gstCamera(width,height,'/dev/video1') # 2.7 fps
webcam2=jetson.utils.gstCamera(width,height,'/dev/video2') # 6.1 fps
picam=jetson.utils.gstCamera(width,height,'0')             # 29.2 fps

net = jetson.inference.imageNet('googlenet')
font = cv2.FONT_HERSHEY_SIMPLEX
StartTime = time.time()
fpsFilter = 0
while True:
#    frame,width,height = webcam1.CaptureRGBA(zeroCopy=1)
#    frame,width,height = webcam2.CaptureRGBA(zeroCopy=1)
    frame,width,height = picam.CaptureRGBA(zeroCopy=1)
    classID,confidence = net.Classify(frame,width,height)
    item=net.GetClassDesc(classID)
    dt = time.time() - StartTime
    fps = 1/dt
    fpsFilter = 0.95*fpsFilter + 0.05*fps
    StartTime = time.time()
    title = str(round(fpsFilter,1)) + ' fps ' + item
    frame = jetson.utils.cudaToNumpy(frame,width,height,4)
    frame = cv2.cvtColor(frame,cv2.COLOR_RGBA2BGR).astype(np.uint8)
    cv2.putText(frame,title,(0,30),font,1,(0,0,255),2)
    cv2.imshow('win',frame)
    cv2.moveWindow('win',0,0)
    
    if cv2.waitKey(1) == ord('q'):
        break

webcam1.release()
webcam2.release()
picam.release()
cv2.destroyAllWindows()

