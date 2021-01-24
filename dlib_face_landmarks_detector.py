import os
import argparse
import dlib
import cv2

def get_argparser():
    parser = argparse.ArgumentParser(description='Skin Lesion Classifier')
    parser.add_argument("--path", nargs='?', type=str, default="",
                        help="Path of the video or image to detect faces in. If Blank, camera feed will be used.")
    return parser.parse_args()

args = get_argparser()


# Press 'Escape' key to quit process.
class face_landmarks_detector():
    def __init__(self, args, cam, detector, predictor):
        self.path = args.path
        self.cam = cam
        self.detector = detector
        self.predictor = predictor

    def detect(self):
        while True:
            if '.jpg' in self.path:
                frame = cv2.imread(self.path)
            else:
                _, frame = self.cam.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)  # detect face

            for i, face in enumerate(faces):  # for all the faces detected in the frame
                x1 = face.left()
                y1 = face.top()
                x2 = face.right()
                y2 = face.bottom()

                # Note: cv2 color scheme is BGR.
                # Detection is done on Grayscale image and mapped back to BGR to display.
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # plot rectangle around the face

                landmarks = self.predictor(gray, face)  # detect 68 landmarks

                for j in range(68):
                    x = landmarks.part(j).x
                    y = landmarks.part(j).y
                    cv2.circle(frame, (x, y), 6, (255, 0, 0), -1)  # plot each of the landmarks with circle

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1)

            if key == 27:
                break


if __name__ == "__main__":
    if args.path == "":
        args.path = 0
    else:
        if not os.path.exists(args.path):
            print("Cannot locate file at path: {}".format(args.path))
        if not os.path.isfile(args.path):
            print("passed path is not a file: {}".format(args.path))

    cam = cv2.VideoCapture(args.path)
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    fld = face_landmarks_detector(args, cam, detector, predictor)

    fld.detect()
  
