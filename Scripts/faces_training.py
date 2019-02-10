import os
import numpy as np
from PIL import Image
import cv2 as cv

face_cascade = cv.CascadeClassifier('./cascades/haarcascade_frontalface_alt2.xml')
recognizer = cv.face.LBPHFaceRecognizer_create()

x_train = []
y_labels = []
path = "./Images"
for folder_name in os.listdir(path):
    for image_name in os.listdir(path+'/'+folder_name):
        if image_name.endswith("jpg") or image_name.endswith("jpeg"):
            total_image_path = path+'/'+folder_name+'/'+image_name
            
            pil_image = Image.open(total_image_path).convert('L')
            image_array = np.array(pil_image,"uint8")
            size = (550,550)
            pil_image = pil_image.resize(size, Image.ANTIALIAS)
            
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                #cv.imshow(folder_name,roi)
                #cv.waitKey(0)
                print (folder_name)
                x_train.append(roi)
                y_labels.append(folder_name)
                
            
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_labels = le.fit_transform(y_labels)


np.save('encoder.npy', le.classes_)

recognizer.train(x_train, y_labels)
recognizer.save('model.yml')