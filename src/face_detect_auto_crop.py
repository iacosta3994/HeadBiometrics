import cv2
import sys
import os
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
left_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_righteye_2splits.xml')


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

def crop_above_eyes(img, mag_xy):
    if img is None:
        print("Image not loaded to crop")
        return None
    #cascades for 3 faicial points
    left_eye = left_eye_cascade.detectMultiScale(img, 1.05, 5)
    right_eye = right_eye_cascade.detectMultiScale(img, 1.05, 5)


    # X Y W H cordinates for each detected feature
    (mag_x, mag_y) = mag_xy
    for (lex, ley, lew, leh) in left_eye:
        for (rex, rey, rew, reh) in right_eye:


            #adapting H W for each image
            height, width = img.shape[:2]

            # nose position is to the right than both eyes
            if mag_x > lex and mag_x > rex:
                if ley > mag_y > rey:
                    above_eyes = ((lex + rex) / 2) - int(leh + reh / 4)
                    beginX = 0
                    endX = above_eyes
                    beginY = 0
                    endY = height
                    top_img = img[beginX:endX, beginY:endY]
                    return top_img
            # Nose is between both eyes and is below it
            elif lex < mag_x < rex:
                if mag_y > lex and mag_y > rex:
                    above_eyes = ((ley + rey) / 2) - int(leh + reh / 4)
                    beginX = 0
                    endX = width
                    beginY = 0
                    endY = above_eyes
                    top_img = img[beginX:endX, beginY:endY]
                    return top_img
            # Nose position is on the left side of both eyes
            elif mag_x < lex and mag_x > rex:
                if rey > mag_y > ley:
                    above_eyes = ((lex + rex) / 2) + int(leh + reh / 4)
                    beginX = above_eyes
                    endX = width
                    beginY = 0
                    endY = height
                    top_img = img[beginX:endX, beginY:endY]
                    return top_img
            # Nose is between both eyes but is above it
            elif lex < mag_x < rex:
                if mag_y < ley and mag_y < rey:
                    above_eyes = ((ley + rey)/2) + int(leh + reh / 4)
                    beginX = 0
                    endX = width
                    beginY = above_eyes
                    endY = height
                    top_img = img[beginX:endX, beginY:endY]
                    return top_img
            else:
                print("could not crop image")
                print((lex, ley, lew, leh), (rex, rey, rew, reh), (mag_x, mag_y) )
                return img
