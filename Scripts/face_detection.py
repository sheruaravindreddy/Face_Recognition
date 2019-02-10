import cv2 as cv
import numpy as np
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")

face_cascade = cv.CascadeClassifier('./cascades/haarcascade_frontalface_alt2.xml')
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read('model.yml')

le = LabelEncoder()
le.classes_ = np.load('encoder.npy')

cap = cv.VideoCapture(0)
while True:
    #...Capture frame by frame
    ret, frame = cap.read()
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    
    #...Detecting faces and saving the final captured frame cropping face
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for x,y,w,h in faces:
        #print (x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        
        #img_item = 'face_detected.jpg'
        #cv.imwrite(img_item, roi_color)
        
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45:
            name = le.inverse_transform(id_)
            print (name, id_)
            font = cv.FONT_HERSHEY_COMPLEX
            text_color = (153,51,255)
            stroke = 2
            cv.putText(frame, name+'--'+str(conf), (x,y), font, 1, text_color, stroke, cv.LINE_AA)
        
        color = (0,128,0)
        stroke = 2
        cv.rectangle(frame, (x,y), (x+w,y+h), color,stroke)
    #...Display the frame
    cv.imshow('frame',frame)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()