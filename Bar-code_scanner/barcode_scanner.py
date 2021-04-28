import cv2
import numpy as np
from pyzbar.pyzbar import decode

# img = cv2.imread('Bar-code_scanner\img.jpg')
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:

    success, img = cap.read()
    code = decode(img)
    for barcode in decode(img):
        mydata = barcode.data.decode('utf-8')
        print(mydata)
        pts = np.array([barcode.polygon], np.int32)
        pts = pts.reshape(-1,1,2)
        cv2.polylines(img, [pts], True, (255,0,255), 5)
        text_pts = barcode.rect
        cv2.putText(img, mydata, (text_pts[0], text_pts[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255), 2)

    cv2.imshow('result', img)
    cv2.waitKey(1)