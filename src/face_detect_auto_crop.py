import cv2
import sys
import os
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


def face_detect_auto_crop(img, save_result):

    if img is None:
        print("Can't open image file")
        return None

    #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(img, 1.1, 3, minSize=(100, 100))
    eye = eye_cascade.detectMultiScale(img,1.1, 3, minSize =(100,100))

    cascade_used = eye

    if cascade_used is None:
        return None


    for (x, y, w, h) in cascade_used:
        r = max(w, h) / 2
        centerx = x + w / 2
        centery = y + h / 2
        nx = int(centerx - r)
        ny = int(centery - r)
        nr = int(r * 2)
        face_img = img[ny:ny+nr, nx:nx+nr]
        final_img = cv2.resize(face_img, (200, 200))
            if save_result:
                for (x, y, w, h) in cascade_used:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.imwrite('img.png', final_img)
        return final_img
