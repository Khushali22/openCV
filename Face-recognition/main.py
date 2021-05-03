import cv2
import numpy as np
import face_recognition

#Loading image 
imgElon = face_recognition.load_image_file('Face-recognition/images/elonMusk_train.jpg')
# convert it into RGB Format
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
#Loading image for test image
imgElon_test = face_recognition.load_image_file('Face-recognition/images/bill-gates-2.jpg')
imgElon_test = cv2.cvtColor(imgElon_test, cv2.COLOR_BGR2RGB)

faceloct = face_recognition.face_locations(imgElon)[0]
#Encoding face
encodeElon = face_recognition.face_encodings(imgElon)[0]
# Drawing rectangle on detected face
cv2.rectangle(imgElon, (faceloct[3], faceloct[0]), (faceloct[1], faceloct[2]), (255,0,255), 2)

# For test image
faceloctTest = face_recognition.face_locations(imgElon_test)[0]
#Encoding face
encodeElonTest = face_recognition.face_encodings(imgElon_test)[0]
cv2.rectangle(imgElon_test, (faceloctTest[3], faceloctTest[0]), (faceloctTest[1], faceloctTest[2]), (255,0,255), 2)


# To match both elon msuk test and original images, use encodings( 128 measurements)
result = face_recognition.compare_faces([encodeElon], encodeElonTest)
# Find similarity between two images/faces here, find distance 
faceDistance = face_recognition.face_distance([encodeElon], encodeElonTest)
print(result, faceDistance)
cv2.putText(imgElon_test, f'{result} {round(faceDistance[0],2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255), 2)

cv2.imshow('ELon musk' , imgElon)
cv2.imshow('ELon musk test' , imgElon_test)
cv2.waitKey(0)
