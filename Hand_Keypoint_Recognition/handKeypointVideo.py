from __future__ import division
import time
import cv2 as cv
import numpy as np


protoFile = "hand/pose_deploy.prototxt"
weightsFile = "hand/pose_iter_102000.caffemodel"
nPoints = 22
POSEPAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[9,10],[10,11],[11,12],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]
PALMPAIRS = [ [0, 5],[5, 9],[9,13],[13, 17], [17, 0] ]
threshold = 0.2
inHeight = 368
net = cv.dnn.readNetFromCaffe(protoFile, weightsFile)


capture = cv.VideoCapture(0)
while True:
    isTrue, frame= capture.read()
    print(frame)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    aspectRatio = frameWidth/frameHeight
    inWidth = int(((aspectRatio*inHeight)*8)//8)
    blob= cv.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(blob)
    output = net.forward()
    points = []

    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]
        probMap = cv.resize(probMap, (frameWidth, frameHeight))

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv.minMaxLoc(probMap)
        print(prob)
        if prob > threshold :
            print("prob")
            # Add the point to the list if the probability is greater than the threshold
            points.append((int(point[0]), int(point[1])))
        else :
            points.append(None)

    # Draw Skeleton
    for pair in POSEPAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv.line(frame, points[partA], points[partB], (0, 255, 255), 2)
            cv.circle(frame, points[partA], 5, (0, 0, 255), thickness=-1, lineType=cv.FILLED)
            cv.circle(frame, points[partB], 5, (0, 0, 255), thickness=-1, lineType=cv.FILLED)

    for pair in PALMPAIRS:
        partA = pair[0]
        partB = pair[1]
        if points[partA] and points[partB]:
            cv.line(frame, points[partA], points[partB], (0, 255, 255), 2)
    cv.imshow('Video', frame)
    if cv.waitKey(20) & 0xFF == ord('d'):
        break
capture.release()
cv.destroyAllWindows() 
cv.waitKey(0)


#frame = cv.imread("Photos/images.jpeg")
#frameCopy = np.copy(frame)



#t = time.time()


