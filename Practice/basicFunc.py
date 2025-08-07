import cv2 as cv
img = cv.imread('Photos/france.jpg')
cv.imshow('paris', img)
#gray scale
grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', grey)

#Blur, ksize has to be odd
blur = cv.GaussianBlur(img, (7, 7), cv.BORDER_DEFAULT)
cv.imshow('blur', blur)
# Edge Cascade
canny = cv.Canny(img, 125, 175)
cv.imshow('Canny Edges', canny)
# image dialting (thicked edges)
dilated = cv.dilate(canny, (3, 3), iterations=1)
cv.imshow('Dilated', dilated)
# Eroding (tries to revert edges back to edge cascade image)
eroded = cv.erode(dilated, (3,3), iterations = 1)
cv.imshow('Eroded', eroded)
 # Resize
 # resizes to 500 by 500 ignoring aspect ration
resized = cv.resize(img, (500, 500))
cv.imshow('resized', resized)
#Cropping
cropped = img[50:200, 200:400]
cv.imshow("cropped", cropped)

cv.waitKey(0)