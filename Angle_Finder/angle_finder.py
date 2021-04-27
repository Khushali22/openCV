import cv2
import math

path = "ANgle_Finder/img1.png"
img = cv2.imread(path)
pointsList = []

def mouse_points(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        size = len(pointsList)
        if size != 0 and size % 3 != 0:
            cv2.line(img, tuple(pointsList[round((size-1 )/3)*3]) ,(x,y), (0,0,255), 2)
        cv2.circle(img, (x, y), 5, (0,0,255), cv2.FILLED)
        pointsList.append([x, y])

def gradients(p1, p2):
    return (p2[1] - p1[1])/ (p2[0] - p1[0]) # SLope or gradient y2-y1/ x2-x1


def getAngle(pointsList):
    p1, p2, p3 = pointsList[-3:]
    m1 = gradients(p1, p2)
    m2 = gradients(p1, p3)
    angleR = math.atan((m2-m1)/(1+(m2*m1)))
    angleD = round(math.degrees(angleR))
    cv2.putText(img, str(angleD), (p1[0]-40, p1[1]-20),cv2.FONT_HERSHEY_COMPLEX, 1.5, (0,0,255),2)

while True:

    if len(pointsList) % 3 == 0 and len(pointsList) != 0:
        getAngle(pointsList)
    cv2.imshow('Image', img)
    cv2.setMouseCallback('Image', mouse_points)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pointsList = []
        img = cv2.imread(path)