import cv2 as cv
import numpy as np
video = cv.VideoCapture(0)
if not video.isOpened():
    raise IOError("Cannot open video capture device.")
prevFrame = np.zeros(video.read()[1].shape[:2], dtype='uint8')
#prevprevFrame = np.zeros(video.read()[1].shape[:2], dtype='uint8')
movement = np.zeros(video.read()[1].shape[:2], dtype='uint8')
blank = np.zeros(video.read()[1].shape[:2], dtype='uint8')
white = cv.imread('Images/blank.png')
while True:
    isTrue, frame = video.read()
    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # print(grey.shape)
    blur = cv.GaussianBlur(grey, (29, 29), 0)
    #edge = cv.Canny(blur, 75, 200)
    ret, thresh = cv.threshold(blur, 0, 255, cv.THRESH_OTSU|cv.THRESH_BINARY)
    difference  = cv.bitwise_xor(thresh, prevFrame)
    #difftemp = cv.bitwise_xor(prevFrame, prevprevFrame)
    #difference = cv.bitwise_xor(difference, difftemp)
    contours, hierarchy = cv.findContours(difference, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cv.putText(difference, f"{len(contours)}", ((int)(difference.shape[0]/2),(int) (difference.shape[1]/2)), cv.FONT_HERSHEY_COMPLEX, 5, 255, 3)
    print(f"________________________\n")
    for c in contours:
        #x, y, w, h = c
        print(cv.contourArea(c))
     
    if (len(contours) < 600):
        movement = cv.bitwise_and(movement, blank)
        cv.putText(movement, "MOVEMENT", ((int)(movement.shape[0]/2),(int) (movement.shape[1]/2)), cv.FONT_HERSHEY_PLAIN, 10, (255, 255, 255), 3)
        cv.circle(white, ((int)(white.shape[0]/2) + 50,(int) (white.shape[1]/2) -50), 50, (0, 0, 255), -1)
    else:
        movement = cv.bitwise_and(movement, blank)
        cv.putText(movement, "NO MOVEMENT", ((int)(movement.shape[0]/2),(int) (movement.shape[1]/2)), cv.FONT_HERSHEY_PLAIN, 10, (255, 255, 255), 3)
        cv.circle(white, ((int)(white.shape[0]/2) + 50,(int) (white.shape[1]/2) -50), 50, (0, 255, 0), -1)
    #print(f"{len(contours)}")
    cv.imshow("Videofeed", difference)
    cv.imshow("Movement", white)
    prevFrame = thresh
    #prevprevFrame = prevFrame
    if cv.waitKey(20) & 0xFF == ord('d'):
        break
video.release()
cv.destroyAllWindows() 
cv.waitKey(0) 