import jetson.inference
import jetson.utils
import time
import numpy as np

timeStamp=time.time()
fpsFiltered=0

net = jetson.inference.detectNet('ssd-mobilenet-v2',threshold=0.5)
dispW=1280
dispH=720

#picam = jetson.utils.gstCamera(dispW,dispH,'0')
webcam1 = jetson.utils.gstCamera(dispW,dispH,'/dev/video1')
webcam2 = jetson.utils.gstCamera(dispW,dispH,'/dev/video2')

display = jetson.utils.glDisplay()
while display.IsOpen():
#    img,width,height = picam.CaptureRGBA()
#    img,width,height = webcam1.CaptureRGBA()
    img,width,height = webcam2.CaptureRGBA()
    dections = net.Detect(img,width,height)
    display.RenderOnce(img,width,height)
    dt = time.time() - timeStamp
    timeStamp=time.time()
    fps = 1/dt
    fpsFiltered = 0.9*fpsFiltered + 0.1*fps
    print('FPS',round(fpsFiltered,1))
