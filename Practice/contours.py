import cv2 as cv
img = cv.imread('Photos/france.jpg')
cv.imshow('paris', img)

gray  = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)
canny = cv.Canny(img, 125, 175)
cv.imshow('Canny edge', canny)

contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
print(len(contours))
print("COntours found")
# A contour is essentially the curves that join to points along a edge
 

cv.waitKey(0)