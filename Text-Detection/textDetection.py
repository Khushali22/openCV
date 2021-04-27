import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

img = cv2.imread("img.jpg")
#convert it into RGB value (OPencv by default in BGR value)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# print(pytesseract.image_to_string(img))

## Detecting characters
# print(pytesseract.image_to_boxes(img))
# height_img , width_img, _ = img.shape
# boxes = pytesseract.image_to_boxes(img)

# for b in boxes.splitlines():
#     b = b.split(" ")
#     x,y,w,h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
#     cv2.rectangle(img, (x,height_img-y), (w,height_img-h), (0,0,255), 2)   # generally we need to do subtraction for width & height but they have correct one for width, so we need to subtract from height
#     cv2.putText(img, b[0], (x, height_img-y+25), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255), 2 )
# cv2.imshow("Result", img)
# cv2.waitKey(0)

# ## Detecting words
# # print(pytesseract.image_to_boxes(img))
# height_img , width_img, _ = img.shape
# boxes = pytesseract.image_to_data(img)
# for x,b in enumerate(boxes.splitlines()):
#     if x!=0:
#         b = b.split()
#         print(b)
#         if(len(b) == 12):
#             x,y,w,h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
#             cv2.rectangle(img, (x,y), (w+x,h+y), (0,0,255), 2)   # generally we need to do subtraction for width & height but they have correct one for width, so we need to subtract from height
#             # cv2.putText(img, b[11], (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255), 2 )
# cv2.imshow("Result", img)
# cv2.waitKey(0)

## Detecting numbers
height_img , width_img, _ = img.shape
configurations = r'--oem 3 --psm 6 outputbase digits'
boxes = pytesseract.image_to_data(img, config = configurations)
for x,b in enumerate(boxes.splitlines()):
    if x!=0:
        b = b.split()
        print(b)
        if(len(b) == 12):
            x,y,w,h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
            cv2.rectangle(img, (x,y), (w+x,h+y), (0,0,255), 2)   # generally we need to do subtraction for width & height but they have correct one for width, so we need to subtract from height
            # cv2.putText(img, b[11], (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255), 2 )
cv2.imshow("Result", img)
cv2.waitKey(0)