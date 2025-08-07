import cv2 as cv

img = cv.imread('Photos/person.jpeg')
#video = cv.VideoCapture(0)

cv.imshow('person', img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)
haar_cascade = cv.CascadeClassifier('haar_face.xml')

faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 3) # Change numbers to tune face dectection

print(f'Number of faces found = {len(faces_rect)}')
for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness = 2)
cv.imshow('dectected faces', img)
#while (True):
#    isTrue, frame=video.read()
#    g = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#    faces = haar_cascade.detectMultiScale(g, 1.1, 3)
#    for (x, y, w, h) in faces:
#        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
#    if cv.waitKey(20) & 0xFF == ord('d'):
#        break
#    cv.imshow('Video', frame)
#video.release()
#cv.destroyAllWindows() 
cv.waitKey(0)