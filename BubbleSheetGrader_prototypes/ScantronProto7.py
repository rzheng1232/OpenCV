import cv2 as cv
import pytesseract as pt
import numpy as np
import pandas as pd
import os
from pathlib import Path
import shutil
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
#from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import argparse
#import imutils
# method that changes the exposure by changing the gamma value for the whole image
def adjust_exposure(image, gamma=1.0):
    invG = 1.0/gamma
    table = np.array([((i/255.0) ** invG)*255 
                     for i in np.arange(0, 256)]).astype("uint8")
    return cv.LUT(image, table)
img = cv.imread('Photos/newScan5.png')
Name = ""
a = os.path.isfile("AnswerKeys/b.csv")
while True:
    Name = input("Name of test: ")
    key = []
    k = 1
    
    if (os.path.isfile(f"AnswerKeys/{Name}.csv")):
        response = input("The test key already exists. Do you want to overwrite ('o'), reuse('r'), or enter a different name('c')? ")
        if (response == 'o'):
            os.remove(f"AnswerKeys/{Name}.csv")
            userIn = input(f"Question {k}: ")
            while (userIn != "STOP"):
                key.append(userIn)
                k += 1
                userIn = input(f"Question {k}: ")
            questionNums = [v for v, k in enumerate(key)]
            answerKey = {' A:':key}
            df = pd.DataFrame(answerKey)
            df.to_csv(f"AnswerKeys/{Name}.csv")
            print(str(df))
            print(key)
            break
            
        elif(response == 'r'):
            break
        elif(response == 'c'):
            continue
            
    else:
        #os.remove(f"AnswerKeys/{Name}.csv")
        userIn = input(f"Question {k}: ")
        while (userIn != "STOP"):
            key.append(userIn)
            k += 1
            userIn = input(f"Question {k}: ")
        questionNums = [v for v, k in enumerate(key)]
        answerKey = {' A:':key}
        df = pd.DataFrame(answerKey)
        df.to_csv(f"AnswerKeys/{Name}.csv")
        print(str(df))
        print(key)
        break
        

cv.imshow('image', img)
img = adjust_exposure(img, gamma=0.2)
cv.imshow('imageEx', img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(gray, (11, 11), 0)
edge = cv.Canny(blurred, 75, 200)
cv.imshow('edge', edge)
cnt = cv.findContours(edge, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
Rcontours = imutils.grab_contours(cnt)
Rcontours = sorted(Rcontours, key = cv.contourArea, reverse = True)[:5]
print (len (Rcontours))
for c in Rcontours:
    perimeter = cv.arcLength(c, True)
    approximation = cv.approxPolyDP(c, 0.02*perimeter, True)
    print(len(approximation))
    if (len(approximation) == 4):
        screenCont = approximation
        break
warped = four_point_transform(gray, screenCont.reshape(4, 2))
#cv.drawContours(img, [screenCont], -1, (0, 255, 0), 2)
cv.imshow("Warped im", warped)
WarpedBlur = cv.GaussianBlur(warped, (5, 5), 0)
WarpedEdge = cv.Canny(WarpedBlur, 75, 200)
ret, thresh = cv.threshold(WarpedEdge, 0, 255, cv.THRESH_OTSU|cv.THRESH_BINARY)
cv.imshow('thresh', thresh)

rectKernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

dilation = cv.dilate(thresh, rectKernel, iterations=1)
cv.imshow('dilate', dilation)
contours, hierarchy = cv.findContours(dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

bRects = [cv.boundingRect(i) for i in contours]

print("                                                                     ")

bRects = sorted(bRects, key=lambda y: y[1])

currRow = []
intensities = []
answers = []
black = np.zeros(warped.shape[:2], dtype='uint8')
Inc = 0
answerPos = []
for i in range(len(contours)):
    x, y, w, h = bRects[i]
    if (w/h > .9 and w/h < 1.1 and w > 25 ) or i == len(contours)-1: 
        if len(currRow) == 0:
            currRow.append(bRects[i])
        else:
            x1, y1, w1, h1 = bRects[i-1]
            
            if abs(y - y1) < 3:
                currRow.append(bRects[i])
            else:
                currRow = sorted(currRow, key=lambda x:x[0])
                for g in currRow:
                    mask = np.zeros(warped.shape[:2], dtype='uint8')
                    Inc+=1
                    x2, y2, w2, h2 = g
                    cx1 = (int)(x2 + w2/2)
                    cy1 = (int)(y2 + h2/2)
                    cv.circle(mask, (cx1, cy1), (int)(w2/2), 255, thickness=-1)
                    masked = cv.bitwise_and(dilation, mask)
                    white = cv.countNonZero(masked)
                    cv.imshow('masked: ' , masked)
                    intensities.append(white)
                    l = f"{Inc}"
                    cv.putText(black, l, (cx1-5, cy1), cv.FONT_HERSHEY_COMPLEX, 0.3, (255, 255,255 ))
                answer = intensities.index(min(intensities))
                answers.append(chr(answer+65))
                answerPos.append(((int)(currRow[answer][0] + currRow[answer][2]/2), (int)(currRow[answer][1] + currRow[answer][3]/2)))                
                currRow = []
                currRow.append(bRects[i])
                intensities.clear()
print(answers)
print(answerPos)

answerSheet = pd.read_csv(f"AnswerKeys/{Name}.csv")
a = answerSheet.values.tolist()
correct = 0
wrong = 0
for i in answerPos:
    x, y = i
    cv.circle(warped, (x, y), 30, (0, 255, 0), 3)
    
'''
for i in range(len(answers)):
    curr = answers[i]
    corr = a[i][1]
    x, y = answerPos[i]
    if (curr == corr):
        cv.circle(warped, (x, y), 30, (0, 255, 0), 3)
        correct+=1
    else:
        cv.circle(warped, (x, y), 30, (0, 0, 255), 3)
        wrong+=1
score = ((correct)/(correct+wrong)) * 100
cv.putText(warped,f"{score}%" , (300, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255,0 ))
print(a)'''
cv.imshow('h', warped)
cv.waitKey(0)
