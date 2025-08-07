import cv2 as cv
import matplotlib.pyplot as plt
img = cv.imread('Photos/person.jpeg')
cv.imshow('person', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Simple thresholding (add _INV to end of tag to invert to black and white)

threshold, thresh = cv.threshold(gray, 180, 255, cv.THRESH_BINARY)
cv.imshow('simple', thresh)

# Adaptive THresholding
adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 15, 3)
cv.imshow('adaptive', adaptive_thresh)

cv.waitKey(0)