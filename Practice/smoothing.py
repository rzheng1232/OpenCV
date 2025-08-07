import cv2 as cv

img = cv.imread('Photos/person.jpeg')
cv.imshow('person', img)

# Averaging (A pixel is set to the average of all its surrounding pixels)
average = cv.blur(img, (3, 3))
cv.imshow('Average', average)
# Gaussian Blur (Average of product of weights of surrounding pixels, more natural but less bluring)
gauss = cv.GaussianBlur(img, (3, 3), 0)
cv.imshow('Gaussian', gauss)
# Median Blur (FInds median of surrounding pixels. Helps reduce noise. )
median = cv.medianBlur(img, 3)
cv.imshow('Median', median)
# Bilateral Blur (Most effective. Blurs, but retains edges)
bilateral = cv.bilateralFilter(img, 10, 35, 25) #Sigma space increases pixels that influence the blurring
cv.imshow("Bilateral ", bilateral)

cv.waitKey(0)