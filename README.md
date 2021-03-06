# Face-Detection-and-Landmark-Detection-with-OpenCV-and-DLib

Code to detect face and the 68 landmark points on face, from the camera feed.  
Supports Image, Video and live camera feed.
Supports two types of face detctors from DLib - HoG & CNN. 

## Usage:  
For Image inputs: python dlib_face_landmarks_detector.py --path="input.jpg" --face_detector_type="hog"  
For Video inputs: python dlib_face_landmarks_detector.py --path="input.mp4" --face_detector_type="cnn"  
For Cameta inputs: python dlib_face_landmarks_detector.py --face_detector_type="cnn"  

Close the window or press 'Escape' key to stop the detection process.  

<!-- File 'shape_predictor_68_face_landmarks.dat' denotes weights file for the landmarks predictor model. -->

## Requirements:  
openCV (4.1.0.25)  
dlib (19.21.1)  

## Sample Output
<img src="/outputs/multi_face_1_output.jpg" width="1000" height="400">  

<img src="/outputs/multi_face_output.jpg" width="700" height="400">

