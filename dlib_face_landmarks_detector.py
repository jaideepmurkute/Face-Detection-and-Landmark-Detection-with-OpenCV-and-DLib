import dlib
import cv2

cam = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Press 'Escape' key to quit process.

while True:
    _, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)  # detect face

    for i, face in enumerate(faces):  # for all the faces detected in the frame
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        # Note: cv2 color scheme is BGR.
        # Detection is done on Grayscale image and mapped back to BGR to display.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # plot rectangle around the face

        landmarks = predictor(gray, face)  # detect 68 landmarks

        for j in range(68):
            x = landmarks.part(j).x
            y = landmarks.part(j).y
            cv2.circle(frame, (x, y), 6, (255, 0, 0), -1)  # plot each of the landmarks with circle

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break