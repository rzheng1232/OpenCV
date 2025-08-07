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
#cv.imshow('imageEx', img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(gray, (5, 5), 0)
edge = cv.Canny(blurred, 75, 200)
ret, thresh = cv.threshold(edge, 0, 255, cv.THRESH_OTSU|cv.THRESH_BINARY)
rectKernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
dilation = cv.dilate(thresh, rectKernel, iterations=1)
cv.imshow('imageEx', dilation)

contours, hierarchy = cv.findContours(dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
bRects = [cv.boundingRect(i) for i in contours]
bRects = sorted(bRects, key=lambda y: y[1])
print(bRects)
'''-------------------------------------------------------------------------------------'''




currRow = []
Intensities = []
answers = []
v = np.zeros(img.shape[:2], dtype='uint8')
Inc = 0
for i in range(len(bRects)):
    
    x, y, w, h = bRects[i]
    cx = (int)(x + w/2)
    cy = (int)(y + h/2)
    if (w/h > .9 and w/h < 1.1 and w > 25) or i == range(len(bRects)):
        if len(currRow)== 0:
            print('first')
            #print(bRects[i-1])

            print(bRects[i])
            currRow.append(bRects[i])
            
        else:
            x1, y1, w1, h1 = bRects[i-1]
            if abs(y - y1) < 3:
                print(bRects[i])
                currRow.append(bRects[i])
            else:
                
                currRow = sorted(currRow, key=lambda x:x[0])
                
                for g in currRow:
                    mask = np.zeros(img.shape[:2], dtype='uint8')
                    
                    Inc+=1
                    x2, y2, w2, h2 = g
                    cx1 = (int)(x2 + w2/2)
                    cy1 = (int)(y2 + h2/2)
                    cv.circle(mask, (cx1, cy1), (int)(w/2), 255, thickness=-1)
                    masked = cv.bitwise_and(dilation, mask)
                    white = cv.countNonZero(masked)
                    print(f"    circle intensity: {white}")
                    Intensities.append(white)
                    l = f"{Inc}"
                    cv.putText(v, l, (cx1-5, cy1), cv.FONT_HERSHEY_COMPLEX, 0.3, (255, 255,255 ))
                answer = Intensities.index(min(Intensities))
                answers.append(chr(answer+65))
                print(Intensities)
                #print (f"Answer: {answer}")
                print(currRow)
                currRow = []
                currRow.append(bRects[i])
                Intensities.clear()
        #i = f"({x}, {y})"
        
        cv.circle(img, (cx, cy), (int)(w/2), (0, 255, 0), thickness=5)
        #masked = cv.bitwise_and(dilation, mask)
        #text = pt.image_to_string(masked)
        #white = cv.countNonZero(masked)
        


currRow = sorted(currRow, key=lambda x:x[0])
                
for g in currRow:
    mask = np.zeros(img.shape[:2], dtype='uint8')
    Inc+=1
    x2, y2, w2, h2 = g
    cx1 = (int)(x2 + w2/2)
    cy1 = (int)(y2 + h2/2)
    cv.circle(mask, (cx1, cy1), (int)(w/2), 255, thickness=-1)
    masked = cv.bitwise_and(dilation, mask)
    white = cv.countNonZero(masked)
    print(f"    circle intensity: {white}")
    Intensities.append(white)
    l = f"{Inc}"
    cv.putText(v, l, (cx1-5, cy1), cv.FONT_HERSHEY_COMPLEX, 0.3, (255, 255,255 ))
answer = Intensities.index(min(Intensities))
answers.append(chr(answer+65))
print(f"Intensities: {Intensities}")
#print (f"Answer: {answer}")
print(currRow)
currRow = []
currRow.append(bRects[i])
Intensities.clear()

cv.imshow('result', img)
cv.imshow('result1', v)
print(answers)
'''prevCenterY = 0
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
cv.imshow('v', v)'''
cv.waitKey(0)