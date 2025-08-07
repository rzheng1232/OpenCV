import cv2 as cv
import pytesseract as pt

img = cv.imread("Photos/newScan.png")
cv.imshow('g', img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(gray, (5, 5), 0)
edged = cv.Canny(blurred, 75, 200)
ret, thresh = cv.threshold(edged, 0, 255, cv.THRESH_OTSU|cv.THRESH_BINARY)
cv.imshow('thresh', thresh)
rectKernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))
dilation = cv.dilate(thresh, rectKernel, iterations=1)
cv.imshow('dil',dilation)
contours, hierarchy = cv.findContours(dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
im2 = thresh.copy()

for c in contours:
    x, y, w, h = cv.boundingRect(c)
    cropped = im2[y: y+h,x: x+w]
    text = pt.image_to_string(im2)
    print(f"|||| w: {w} h: {h}")
    cv.rectangle(im2, (x, y), (x+w, y+h), (0,255,0 ), 2)
    cv.imshow('gg', im2)
    print(text + " b")
cv.waitKey(0)