import cv2
import sys
import os
import numpy as np

face_cascade =  cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def face_detect_auto_crop(img, show_result):

    if  img is None:
        print("Can't open image file")
        return 0

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.1, 3, minSize=(100, 100))
    if faces is None:
        print('Failed to detect face')
        return 0

    if show_result:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    facecnt = len(faces)
    print("Detected faces: %d" % facecnt)

    height, width = img.shape[:2]

    for (x, y, w, h) in faces:
        r = max(w, h) / 2
        centerx = x + w / 2
        centery = y + h / 2
        nx = int(centerx - r)
        ny = int(centery - r)
        nr = int(r * 2)

        faceimg = img[ny:ny+nr, nx:nx+nr]
        lastimg = cv2.resize(faceimg, (32, 32))
    return lastimg
