import jetson.inference
import jetson.utils
import time

width = 1280
height = 720

webcam1=jetson.utils.gstCamera(width,height,'/dev/video1') # 2.7 fps
webcam2=jetson.utils.gstCamera(width,height,'/dev/video2') # 6.1 fps
picam=jetson.utils.gstCamera(width,height,'0')             # 29.2 fps

display = jetson.utils.glDisplay()
net = jetson.inference.imageNet('googlenet')
font = jetson.utils.cudaFont()
StartTime = time.time()
fpsFilter = 0
while display.IsOpen():
#    frame,width,height = webcam1.CaptureRGBA()
    frame,width,height = webcam2.CaptureRGBA()
#    frame,width,height = picam.CaptureRGBA()
    classID,confidence = net.Classify(frame,width,height)
    item=net.GetClassDesc(classID)
    dt = time.time() - StartTime
    fps = 1/dt
    fpsFilter = 0.95*fpsFilter + 0.05*fps
    StartTime = time.time()
    title = str(round(fpsFilter,1)) + ' fps ' + item
    t2 = str(round(display.GetFPS(),1)) + 'GetFPS'
    font.OverlayText(frame,width,height,title,5,5,font.Magenta,font.Blue)
    font.OverlayText(frame,width,height,t2,5,40,font.Magenta,font.White)
    display.SetTitle(t2 + 'Title')
    display.RenderOnce(frame,width,height)
    print('Display Width,Height',display.GetWidth(),display.GetHeight())
    print('Frame   Width,Height',width,height)
