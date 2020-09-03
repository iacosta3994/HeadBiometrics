import cv2
import sys


def face_detect_auto_crop(image):


    image_gray = cv2.imread(image, 0)


    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        image_gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )

    print("[INFO] Found {0} Faces!".format(len(faces)))

    for (x, y, w, h) in faces:
        cv2.rectangle(image_gray, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return crop_image
