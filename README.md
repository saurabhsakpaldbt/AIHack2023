# aihack2023
AI Hack Repo

## Introduction
This is an aged care monitoring application that utilizes YOLOv3 for human detection, Deep SORT for human tracking, and face_recognition for face recognition. Additionally, our application incorporates MediaPipe for hand gesture recognition, allowing an emergency call alert to be triggered by applying the gesture pattern 'open palm, close palm, open palm, close palm.'<br><br>
The application is designed to detect abnormal activities both within and outside of the designated time period, providing status alerts accordingly.<br>

https://youtu.be/LooFLqZOdIo<br><br>

## Installation
### Python
Python version 3.10.13<br><br>

### Dependencies
To install required dependencies run:
```
pip install -r requirements.txt
```

### Download Pre-trained Weight File
Put this pre-trained weight file under folder "models/yolov3"<br>
https://pjreddie.com/media/files/yolov3.weights<br><br>

### Face Recognition Images
Put images in "faces" folder<br>
File name for Patient: [name].[file extension]<br>
e.g. Peter.jpg<br><br>
File name for Staff: [name]_staff.[file extension]<br>
e.g. Mary_staff.jpg<br><br>

### Connecting to MongoDB
Rename the sample_ini to .ini.<br>
Place in DB_URI and DB_NAME parameter under .ini<br>
The 'detections' collection is used for storing detection documents.<br><br>

## Running the Application
```
python app.py
```

When the server is ready, visit http://127.0.0.1:8000 in a browser.<br><br>

## References
Bewley, A., Ge, Z., Ott, L., Ramos, F.,& Upcroft, B. (2016). Simple online and realtime tracking.<br>
https://github.com/abewley/sort<br><br>
Wojke, N. & Bewley, A. (2018) Deep Cosine Metric Learning for Person Re-identification.<br>
https://github.com/nwojke/deep_sort<br><br>
Redmon, J., & Farhadi, A. (2018). YOLOv3: An Incremental Improvement.<br>
https://pjreddie.com/yolo/<br><br>
Facial recognition<br>
https://github.com/ageitgey/face_recognition<br><br>
Real-time Hand Gesture Recognition using TensorFlow & OpenCV<br>
https://techvidvan.com/tutorials/hand-gesture-recognition-tensorflow-opencv/