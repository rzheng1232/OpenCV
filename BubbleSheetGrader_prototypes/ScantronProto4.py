import cv2 as cv
import pytesseract as pt
import numpy as np
def adjust_exposure(image, gamma=1.0):
    invG = 1.0/gamma
    table = np.array([((i/255.0) ** invG)*255 
                     for i in np.arange(0, 256)]).astype("uint8")
    return cv.LUT(image, table)
    
img = cv.imread('Photos/newScan.png')
cv.imshow('image', img)
img = adjust_exposure(img, gamma=0.2)
cv.imshow('imageEx', img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(gray, (5, 5), 0)
#cv.imshow('blur', blurred)
edge = cv.Canny(blurred, 75, 200)
#cv.imshow('Canny', edge)
ret, thresh = cv.threshold(edge, 0, 255, cv.THRESH_OTSU|cv.THRESH_BINARY)
cv.imshow('thresh', thresh)
rectKernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
dilation = cv.dilate(thresh, rectKernel, iterations=1)
cv.imshow('dilate', dilation)
contours, hierarchy = cv.findContours(dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
bRects = [cv.boundingRect(i) for i in contours]
prevCenterY = 0
prevCenterX = 0


redInc = 0
greenInc = 125
Inc = 0
row = []
rowI = 0
v = np.zeros(img.shape[:2], dtype='uint8')
first = True

for c in bRects:
    mask = np.zeros(img.shape[:2], dtype='uint8')
    
    #x, y, w, h = cv.boundingRect(c)
    x, y, w, h =  c
    if w/h > .9 and w/h < 1.1 and w > 25:
        cx = (int)(x + w/2)
        cy = (int)(y + h/2)
        
        if abs(cy - prevCenterY) > 2 and not first:
            rowI +=1

            print(f" row {rowI}")

            cv.putText(v, "new row", (cx-5, cy), cv.FONT_HERSHEY_COMPLEX, 0.3, 255)

            redInc = redInc + 50
            greenInc = greenInc + 25
            row = sorted(row, key=lambda x: x[0])
            print(row)
            for b in row:
                Inc += 1
                x2, y2, w2, h2 = b
                l = f"{Inc}"
                xc = (int)(x2 + w2/2)
                yc = (int)(y2 + h2/2)
                print(Inc)
                cv.putText(v, l, (xc-5, yc), cv.FONT_HERSHEY_COMPLEX, 0.3, 255)
                #cv.circle(v, (xc, yc), (int)(w/2), 255, thickness=-1)

            row.clear()
            row.append(c)

            print(redInc)
        else:
            #print(c)
            row.append(c)
            first = False
            
        cv.circle(img, (cx, cy), (int)(w/2), (0, greenInc, redInc), thickness=5)
        #i = f"({x}, {y})"
        #i = f"{Inc}"
        #cv.putText(img, i, (cx-5, cy), cv.FONT_HERSHEY_COMPLEX, 0.3, (0, 255,255 ))
        #cv.addText(img, "Hi", (cx,cy), cv.FONT_HERSHEY_COMPLEX)
        cv.circle(img, (91, 300), 20, 0, -1)
        cv.circle(mask, (cx, cy), (int)(w/2), 255, thickness=-1)
        #cv.circle(blank, (cx, cy), (int)(w/2)+1, 0, thickness=12)
    
        masked = cv.bitwise_and(dilation, mask)
        text = pt.image_to_string(masked)
        white = cv.countNonZero(masked)
        print(f"    circle intensity: {white}")
        cv.imshow('blank', mask)
        cv.imshow('masked', masked)
        prevCenterX = cx
        prevCenterY = cy

cv.imshow('result', img)
cv.imshow('v', v)
cv.waitKey(0)