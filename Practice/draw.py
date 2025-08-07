import cv2 as cv
import numpy as np

blank = np.zeros((500, 500, 3), dtype='uint8')
cv.imshow('Blank', blank)
# Paint image a certain colour
# Reference all pixels and set to green
blank[:] = 0, 255, 0
# Reference pixels w to x, y to z
blank[200:300, 300:400] = 255, 0, 0
# area to draw in, two corners, clor, thickness (can be set as -1 to fill)
cv.rectangle(blank, (0,0), (250,250), (0, 25, 0), thickness = 2)
# Circle
cv.circle(blank, (200, 300), 40, (0, 0, 255), thickness = 3)
# line
cv.line(blank, (100, 230), (50, 400), (255, 255, 255), thickness=3)
# Write text (point is where word is written from)
cv.putText(blank, 'hello', (225, 225), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 100), 2)
cv.imshow('Green', blank)
cv.waitKey(0)