import cv2
import sys
import os
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

left_eye_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')

right_eye_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')

nose_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_nose.xml')

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

def crop_above_eyes(img):
    if img is None:
        print("Can not use image")
        return None

    left_eye = left_eye_cascade.detectMultiScale(img, 1.3, 5)
    right_eye = right_eye_cascade.detectMultiScale(img, 1.3, 5)
    nose = nose_cascade.detectMultiScale(img, 1.3, 5)

    (lex, ley, lew, leh) = left_eye
    (rex, rey, rew, reh) = right_eye
    (nosex, nosey, nosew, noseh) = nose

    width, height = img.shape[:2]

    if nosex > lex and nosex > rex:
        if ley > nosey > rey:
            above_eyes = ((lex + rex) / 2) - int(leh + reh / 4)
            beginX = 0
            endX = above_eyes
            beginY = 0
            endY = height
            top_img = img[beginX:endX, beginY:endY]
            return top_img
    if lex < nosex < rex:
        if nosey > lex and nosey > rex:
            above_eyes = ((ley + rey) / 2) - int(leh + reh / 4)
            beginX = 0
            endX = width
            beginY = 0
            endY = above_eyes
            top_img = img[beginX:endX, beginY:endY]
            return top_img
    if nosex < lex and nosex > rex:
        if rey > nosey > ley:
            above_eyes = ((lex + rex) / 2) + int(leh + reh / 4)
            beginX = above_eyes
            endX = width
            beginY = 0
            endY = height
            top_img = img[beginX:endX, beginY:endY]
            return top_img
    if lex < nosex < rex:
        if nosey < ley and nosey < rey:
            above_eyes = ((ley + rey)/2) + int(leh + reh / 4)
            beginX = 0
            endX = width
            beginY = above_eyes
            endY = height
            top_img = img[beginX:endX, beginY:endY]
            return top_img
