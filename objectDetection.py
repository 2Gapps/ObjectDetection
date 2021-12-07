import cv2
import numpy as np
  
frameWidth = 640
frameHeight = 480

# Camera 
cap = cv2.VideoCapture(0)

# cap = cv2.VideoCapture("Resources/testVideo.mp4")
cap.set(3, frameWidth)
cap.set(4, frameHeight)

# Picture
# img_path = "Resources/testImage.jpg"
# img = cv2.imread("Resources/testImage.jpg")

classFile = "src/coco.names"
configPath = "src/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "src/frozen_inference_graph.pb"

classNames = []
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def objectDetection(img, objetcID=None, show=False):
    detections = []
    classId, confs, bbox = net.detect(img, confThreshold=0.5)
    
    for classId, confs, bbox in zip(classId.flatten(), confs.flatten(), bbox):
        if objetcID == (classId - 1):
            detections.append([classNames[classId - 1], bbox.tolist()])
            
            if show:
                x, y, w, h = bbox
                cv2.rectangle(img, bbox, color=(0, 255, 0), thickness=2) 
                cv2.putText(img, classNames[classId -1].upper(), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
    return detections

while True:
    success, img = cap.read()
    
    detections = objectDetection(img, objetcID=0, show=True)
    
    
    cv2.imshow("Video Streaming", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
