import cv2
import numpy as np

threshold=0.5 # Threshold value to detect object
nms_threshold = 0.2 # supress ration , supress percentage threshold
# img = cv2.imread('Object_detection_mobilenetssd/lena.png')
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10,70)
classNames = []
with open('Object_detection_mobilenetssd/coco.names', 'r') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'Object_detection_mobilenetssd/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath = 'Object_detection_mobilenetssd/frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightPath, configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    classIds, conf, bbox = net.detect(img, confThreshold=threshold)
    bbox = list(bbox)
    conf = list(np.array(conf).reshape(1, -1)[0])
    conf = list(map(float, conf))
    # print(bbox, type(bbox))

    indices = cv2.dnn.NMSBoxes(bbox, conf, threshold, nms_threshold)
    # print(indices)

    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x,y), (x+w, y+h), color = (0,255,0), thickness=5)
        cv2.putText(img, classNames[classIds[i][0]-1].upper(), (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_COMPLEX,1, (0,255,0), 2)

    # if len(classIds) != 0: 
    #     for classId, confidence, box in zip(classIds.flatten(), conf.flatten(), bbox):
    #         cv2.rectangle(img, box, color = (0,255,0), thickness=5)
    #         cv2.putText(img, classNames[classId - 1].upper(), (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_COMPLEX,1, (0,255,0), 2)
    #         cv2.putText(img, str(round(confidence*100,2)) ,(box[0]+200,box[1]+30), cv2.FONT_HERSHEY_COMPLEX,1, (0,255,0), 2)
    cv2.imshow('result', img)
    cv2.waitKey(1)