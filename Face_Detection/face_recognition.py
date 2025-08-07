import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')
people = ['Dwayne Johnson', 'Ryan Zheng']

features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy', allow_pickle=True)
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread("Faces/Ryan Zheng/IMG_6103.jpg")
#img = cv.imread("Faces/Dwayne Johnson/Dwayne1.jpg")
#img = cv.imread("Photos/Ryan.jpg")
video = cv.VideoCapture(0)
#gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#faces_rect = haar_cascade.detectMultiScale(gray, 1.3, 3)

#for (x, y, w, h) in faces_rect:
#    faces_roi = gray[y:y+h, x:x+h]
#    label, confidence = face_recognizer.predict(faces_roi)
#    print(label)
#    print(confidence)
#    print(f'label = {people[label]}, confidence of {confidence}')
#    cv.putText(img, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255,0), thickness=2)
#    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
#cv.imshow('dectected face', img)

#--------------------------
while True:
    isTrue, frame= video.read()
    grayv = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces_rectv = haar_cascade.detectMultiScale(grayv, 1.3, 3)
    for (x, y, w, h) in faces_rectv:
        faces_roiv = grayv[y:y+h, x:x+h]
        labelv, confidencev = face_recognizer.predict(faces_roiv)
        print(labelv)
        print(confidencev)
        print(f'label = {people[labelv]}, confidence of {confidencev}')
        cv.putText(frame, str(people[labelv]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255,0), thickness=2)
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
    cv.imshow('Video', frame)
    if cv.waitKey(20) & 0xFF == ord('d'):
        break
video.release()
cv.destroyAllWindows() 
cv.waitKey(0)
cv.waitKey(0)