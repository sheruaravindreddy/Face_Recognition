import cv2 as cv
import os

user_name = input("Enter the name of the person sitting in front of webcam : ")
image_name = input("Save the image as inside the folder as : ")
if not os.path.exists('./Images/'+user_name):
    os.mkdir('./Images/'+user_name)

face_cascade = cv.CascadeClassifier('./cascades/haarcascade_frontalface_alt2.xml')

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
        
        img_item = './Images/'+user_name+ '/' + image_name +'.jpg'
        cv.imwrite(img_item, roi_color)
        
        color = (0,128,0)
        stroke = 2
        cv.rectangle(frame, (x,y), (x+w,y+h), color,stroke)
    #...Display the frame
    cv.imshow('frame',frame)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()