import cv2 as cv
import numpy as np
video = cv.VideoCapture(0)
prevFrame = np.zeros(video.read()[1].shape[:2], dtype='uint8')
#prevprevFrame = np.zeros(video.read()[1].shape[:2], dtype='uint8')
movement = np.zeros(video.read()[1].shape[:2], dtype='uint8')
blank = np.zeros(video.read()[1].shape[:2], dtype='uint8')

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
    if (len(contours) < 300):
        movement = cv.bitwise_and(movement, blank)
        cv.putText(movement, "MOVEMENT", ((int)(movement.shape[0]/2),(int) (movement.shape[1]/2)), cv.FONT_HERSHEY_PLAIN, 10, (255, 255, 255), 3)
    else:
        movement = cv.bitwise_and(movement, blank)
        cv.putText(movement, "NO MOVEMENT", ((int)(movement.shape[0]/2),(int) (movement.shape[1]/2)), cv.FONT_HERSHEY_PLAIN, 10, (255, 255, 255), 3)
    #print(f"{len(contours)}")
    cv.imshow("Videofeed", difference)
    cv.imshow("Movement", movement)
    prevFrame = thresh
    #prevprevFrame = prevFrame
    if cv.waitKey(20) & 0xFF == ord('d'):
        break
video.release()
cv.destroyAllWindows() 
cv.waitKey(0) 