import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'face-recognition\img_attendance'
images = []
classNames = []

mylist = os.listdir(path)
print(mylist)

for cl in mylist:
    curImg = cv2.imread(f'{path}\{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(imgs):
    encodeList = []
    for img in imgs:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoding = face_recognition.face_encodings(img)[0]
        encodeList.append(encoding)
    
    return encodeList

def markAttendance(name):
    with open('Face-recognition/attendance.csv' , 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            n = line.split(",")
            nameList.append(n[0])
        if name not in nameList:
            now = datetime.now()
            dateFormat = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name} {dateFormat}')


encodedTotalList = findEncodings(images)
print("Encoding done")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img_resize = cv2.resize(img, (0,0), None, 0.25, 0.25)
    img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
    faceCurrentFrame = face_recognition.face_locations(img_resize)
    encodingCurrentFrame = face_recognition.face_encodings(img_resize, faceCurrentFrame)

    for encodeface, faceLocation in zip(encodingCurrentFrame, faceCurrentFrame):
        matches = face_recognition.compare_faces(encodedTotalList, encodeface)
        face_distance = face_recognition.face_distance(encodedTotalList, encodeface)
        matchIndex = np.argmin(face_distance)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLocation
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4  # Because we have resize originally to 0.25 % So we need to scale back to real size 
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0,255,0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255), 2)
            markAttendance(name)

    cv2.imshow('WEbcam', img)
    cv2.waitKey()



# Drawing rectangle on detected face
# cv2.rectangle(imgElon, (faceloct[3], faceloct[0]), (faceloct[1], faceloct[2]), (255,0,255), 2)