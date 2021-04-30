import cv2
import os

path = "panorama_generator/Images"

myFolders = os.listdir(path)
print(myFolders)

for folders in myFolders:
    path1 = path + '/' + folders
    images = []
    myList = os.listdir(path1)
    print("Total no of images detected", len(myList))

    for imgN in myList:
        curImg = cv2.imread(path1+'/'+imgN)
        curImg = cv2.resize(curImg, (0,0), None, 0.2, 0.2)
        images.append(curImg)

    joiner = cv2.Stitcher.create()
    (status, result) = joiner.stitch(images)
    if status == cv2.STITCHER_OK:
        print("Panorama generated")
        cv2.imshow(folders, result)
        cv2.waitKey(1)
    else:
        print("Panorama unsuccessful")

cv2.waitKey(0)