# Face_Recognition
Designing a pipeline for capturing images using webcam. Detection of face using from the image and building a model for Image Recognition


Face Recognition task is accomplished using **CascadeClassifier** and **LBPHFaceRecognizer** of OpenCV. 
*Requirements:* OpenCV, numpy, pillow and sklearn

“Images” folder contains folders (each person’s name) which contains respective images. They are used to label further in our scripts.


**_Step-1_**

Images can be given as input in 2 ways:
1.	Webcam: Run capture_faces.py script which requests for person name (whose folder would be created inside Images folder) and image_name (to save the image inside the folder created)
2.	Create folder inside Images folder with the person’s name and then save images into it.
*Example:* If we need to need to cross verify Suresh (vs) Ramesh, create 2 folders Suresh and Ramesh inside Images folder and then upload pics into those folders respectively.


**_Step-2_**

Run the script faces_training.py
Training the model and saving it. Faces are detected using frontal_cascade of Cascade Classifier and cropped as rectangles using OpenCV. The cropped rectangle is read using PIL and converted into a numpy array. Independent Features(x_train) would be numpy array and Dependent features(y_label) would be the folder name.
model and y_label encoder are saved to the local system.


**_Step-3_**

Run the script face_detection.py
Model saved earlier is loaded and the labels are decoded using the encoder loaded. Thereby we detect the person and display the person’s name along with the confidence percentage.
 

<img src="https://github.com/sheruaravindreddy/Face_Recognition/blob/master/sample_result.png" width="200" height="200" />

