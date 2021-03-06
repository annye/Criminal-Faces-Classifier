import cv2
import glob
import sys


def set_trace():
    """A Poor mans break point
       without this in iPython debugger can generate strange characters.
    """
    from IPython.core.debugger import Pdb
    Pdb().set_trace(sys._getframe().f_back)


faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

# Define emotions
emotions = ["neutral",
            "anger",
            "contempt",
            "disgust",
            "fear",
            "happy",
            "sadness",
            "surprise"]


def detect_faces(emotion):
    # Get list of all images with emotion
    files = glob.glob("sorted_set\\%s\\*" % emotion)

    filenumber = 0
    for f in files:
        # Open image
        frame = cv2.imread(f)
        # Convert image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect face using 4 different classifiers
        face = faceDet.detectMultiScale(gray,
                                        scaleFactor=1.1,
                                        minNeighbors=10,
                                        minSize=(5, 5),
                                        flags=cv2.CASCADE_SCALE_IMAGE)
        face_two = faceDet_two.detectMultiScale(gray,
                                                scaleFactor=1.1,
                                                minNeighbors=10,
                                                minSize=(5, 5),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
        face_three = faceDet_three.detectMultiScale(gray,
                                                    scaleFactor=1.1,
                                                    minNeighbors=10,
                                                    minSize=(5, 5),
                                                    flags=cv2.CASCADE_SCALE_IMAGE)
        face_four = faceDet_four.detectMultiScale(gray,
                                                  scaleFactor=1.1,
                                                  minNeighbors=10,
                                                  minSize=(5, 5),
                                                  flags=cv2.CASCADE_SCALE_IMAGE)

        # Go over detected faces, stop at first detected face, return empty if no face.
        if len(face) == 1:
            facefeatures = face
        elif len(face_two) == 1:
            facefeatures = face_two
        elif len(face_three) == 1:
            facefeatures = face_three
        elif len(face_four) == 1:
            facefeatures = face_four
        else:
            facefeatures = ""

        # Cut and save face
        # get coordinates and size of rectangle containing face
        for (x, y, w, h) in facefeatures:
            print "face found in file: %s" % f
            # Cut the frame to size
            gray = gray[y:y + h, x:x + w]
            try:
                # Resize face so all images have same size
                out = cv2.resize(gray, (350, 350))
                # Write image
                cv2.imwrite("dataset\\%s\\%s.jpg" % (emotion, filenumber), out)
            except:
                # If error, pass file
                pass
        # Increment image number
        filenumber += 1


def main():
    for emotion in emotions:
        # Call our face detection module
        detect_faces(emotion)

if __name__ == '__main__':
    main()
