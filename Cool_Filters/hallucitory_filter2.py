import cv2 as cv
import numpy as np
from collections import deque

video = cv.VideoCapture(0)#'Images/test2.mp4')
prevFrame = np.zeros(video.read()[1].shape[:3], dtype='uint8')
old = np.zeros(video.read()[1].shape[:3], dtype='uint8')
previousFrames = deque()
# Create a red overlay
red = np.zeros(video.read()[1].shape[:3], dtype='uint8')
red[:,:] = (0, 0, 255)

for i in range(20):
    previousFrames.append(np.zeros(video.read()[1].shape[:3], dtype='uint8'))

old = video.read()[1]
#movement = np.zeros(video.read()[1].shape[:2], dtype='uint8')
#blank = np.zeros(video.read()[1].shape[:3], dtype='uint8')
print(len(previousFrames))

while True:
    isTrue, frame = video.read()
    #grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #blur = cv.GaussianBlur(frame, (29, 29), 0)
    # assert blur.dtype == prevFrame.dtype
    # #assert blur.dtype == prevprevFrame.dtype
    # difference = cv.subtract(blur, prevFrame)
    # alpha = 0.75
    # alpha_mask = (difference[:, :, 3] / 255.0) * alpha if difference.shape[2] == 4 else alpha
    if frame is not None:
        blended_image = frame.copy()#cv.addWeighted(frame, 1 - alpha_mask, difference[:, :, :3], alpha_mask, 0)
    
    # difference = cv.subtract(blur, prevprevFrame)
    # alpha = 0.5
    # alpha_mask = (difference[:, :, 3] / 255.0) * alpha if difference.shape[2] == 4 else alpha
    # blended_image = cv.addWeighted(frame, 1 - alpha_mask, difference[:, :, :3], alpha_mask, 0)
    
    for framenumber in range(len(previousFrames)):
        if framenumber%10 == 0:
            index = framenumber#len(previousFrames)-1 - framenumber
            assert frame.dtype == previousFrames[index].dtype
            if (index == 0):
                difference = cv.absdiff(frame, previousFrames[index])
            else:
                difference = cv.absdiff(old, previousFrames[index])#previousFrames[len(previousFrames)-1], previousFrames[index])
            normalized_diff = cv.normalize(difference, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
            alpha = 0.5
            alpha_mask = (red[:, :, 3] / 255.0) * alpha if red.shape[2] == 4 else alpha
            #difference = cv.addWeighted(difference, alpha, red, alpha_mask, 0)
            cv.imshow("red", difference)
            #alpha = 1#-framenumber * 0.1
            
            alpha = 1#(1- framenumber * 0.1)  # Adjust alpha for layering effect
            alpha = max(0, min(1, alpha))  # Ensure alpha is within [0, 1]

            alpha_mask = (difference[:, :, 3] / 255.0)*4 * alpha if difference.shape[2] == 4 else alpha
            blended_image = cv.addWeighted(blended_image, alpha, difference, alpha, 0)

            previousFrames.popleft()
            previousFrames.append(frame)
    cv.imshow("Videofeed", blended_image)
    
    #cv.imshow("Movement", movement)
    #prevFrame = blur
    #prevprevFrame = prevFrame
    if cv.waitKey(20) & 0xFF == ord('d'):
        break
video.release()
cv.destroyAllWindows() 
cv.waitKey(0) 