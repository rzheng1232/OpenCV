from __future__ import division
import cv2 as cv
import time
import numpy as np

protoFile = "hand/pose_deploy.prototxt"
weightsFile = "hand/pose_iter_102000.caffemodel"
nPoints = 22
POSEPAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]
net = cv.dnn.readNetFromCaffe(protoFile, weightsFile)

frame = cv.imread("Photos/hand3.jpg")
frameCopy = np.copy(frame)
frameWidth = frame.shape[1]
frameHeight = frame.shape[0]
aspect_ratio = frameWidth/frameHeight

threshold = 0.1

t = time.time()
# input image dimensions for the network
inHeight = 368
inWidth = int(((aspect_ratio*inHeight)*8)//8)
inpBlob = cv.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

net.setInput(inpBlob)

output = net.forward()
print("time taken by network : {:.3f}".format(time.time() - t))

# Empty list to store the detected keypoints
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
        cv.circle(frameCopy, (int(point[0]), int(point[1])), 2, (0, 255, 255), thickness=-1, lineType=cv.FILLED)
        cv.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv.LINE_AA)
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


cv.imshow('KeyPoints', frameCopy)
cv.imshow('Skeleton', frame)
cv.waitKey(0)