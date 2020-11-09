import cv2
import sys
import os
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
left_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')


def face_detect_auto_crop(img, save_result):

    if img is None:
        print("Can't open image file")
        return None

    #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(img, 1.1, 3, minSize=(100, 100))
    eye = eye_cascade.detectMultiScale(img, 1.1, 3, minSize=(100, 100))

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
    # cascades for left amd right eye
    left_eye = left_eye_cascade.detectMultiScale(img, 1.05, 5)
    right_eye = right_eye_cascade.detectMultiScale(img, 1.05, 5)
    img_candidates = []
    img_orientation = []
    if len(left_eye) > 0 and len(right_eye) > 0:
        # X Y W H cordinates for each eye, Magstripe brought from get Magstripe
        (mag_x, mag_y) = mag_xy
        for (lex, ley, lew, leh) in left_eye:
            for (rex, rey, rew, reh) in right_eye:

                # adapting H W for each image
                height, width = img.shape[:2]

                # Magstripe position is to the right than both eyes
                if mag_x > lex and mag_x > rex:
                    if ley > mag_y > rey:
                        above_eyes = round(((lex + rex) / 2) - (leh + reh / 4))
                        beginX = 0
                        endX = above_eyes
                        beginY = 0
                        endY = height
                        top_img = img[beginY:endY, beginX:endX]
                        img_candidates.append(top_img)
                        img_orientation.append([beginY,endY, beginX,endX])
                # Magstripe is between both eyes and is below it
                elif lex < mag_x < rex:
                    if mag_y > lex and mag_y > rex:
                        above_eyes = round(((ley + rey) / 2) - (leh + reh / 4))
                        beginX = 0
                        endX = width
                        beginY = 0
                        endY = above_eyes
                        top_img = img[beginY:endY, beginX:endX]
                        img_candidates.append(top_img)
                        img_orientation.append([beginY,endY, beginX,endX])

                # Magstripe position is on the left side of both eyes
                elif mag_x < lex and mag_x > rex:
                    if rey > mag_y > ley:
                        above_eyes = round(((lex + rex) / 2) + (leh + reh / 4))
                        beginX = above_eyes
                        endX = width
                        beginY = 0
                        endY = height
                        top_img = img[beginY:endY, beginX:endX]
                        img_candidates.append(top_img)
                        img_orientation.append([beginY,endY, beginX,endX])
                # Magstripe is between both eyes but is above it
                elif lex < mag_x < rex:
                    if mag_y < ley and mag_y < rey:
                        above_eyes = round(((ley + rey)/2) + (leh + reh / 4))
                        beginX = 0
                        endX = width
                        beginY = above_eyes
                        endY = height
                        top_img = img[beginY:endY, beginX:endX]
                        img_candidates.append(top_img)
                        img_orientation.append([beginY,endY, beginX,endX])
                else:
                    print("could not crop image L5")
                    print((lex, ley, lew, leh), (rex, rey, rew, reh), (mag_x, mag_y))
                    return img
    elif len(left_eye) == 0 or len(right_eye) == 0:
        left_eye = left_eye_cascade.detectMultiScale(img, 1.05, 4)
        right_eye = right_eye_cascade.detectMultiScale(img, 1.05, 4)
        if len(left_eye) > 0 and len(right_eye) > 0:
            (mag_x, mag_y) = mag_xy
            for (lex, ley, lew, leh) in left_eye:
                for (rex, rey, rew, reh) in right_eye:

                    # adapting H W for each image
                    height, width = img.shape[:2]

                    # Magstripe position is to the right than both eyes
                    if mag_x > lex and mag_x > rex:
                        if ley > mag_y > rey:
                            above_eyes = round(((lex + rex) / 2) - (leh + reh / 4))
                            beginX = 0
                            endX = above_eyes
                            beginY = 0
                            endY = height
                            top_img = img[beginY:endY, beginX:endX]
                            img_candidates.append(top_img)
                            img_orientation.append([beginY,endY, beginX,endX])
                    # Magstripe is between both eyes and is below it
                    elif lex < mag_x < rex:
                        if mag_y > lex and mag_y > rex:
                            above_eyes = round(((ley + rey) / 2) - (leh + reh / 4))
                            beginX = 0
                            endX = width
                            beginY = 0
                            endY = above_eyes
                            top_img = img[beginY:endY, beginX:endX]
                            img_candidates.append(top_img)
                            img_orientation.append([beginY,endY, beginX,endX])
                    # Magstripe position is on the left side of both eyes
                    elif mag_x < lex and mag_x > rex:
                        if rey > mag_y > ley:
                            above_eyes = round(((lex + rex) / 2) + (leh + reh / 4))
                            beginX = above_eyes
                            endX = width
                            beginY = 0
                            endY = height
                            top_img = img[beginY:endY, beginX:endX]
                            img_candidates.append(top_img)
                            img_orientation.append([beginY,endY, beginX,endX])
                    # Magstripe is between both eyes but is above it
                    elif lex < mag_x < rex:
                        if mag_y < ley and mag_y < rey:
                            above_eyes = round(((ley + rey)/2) + (leh + reh / 4))
                            beginX = 0
                            endX = width
                            beginY = above_eyes
                            endY = height
                            top_img = img[beginY:endY, beginX:endX]
                            img_candidates.append(top_img)
                            img_orientation.append([beginY,endY, beginX,endX])
                    else:
                        print("could not crop image L4")
                        print((lex, ley, lew, leh), (rex, rey, rew, reh), (mag_x, mag_y))
                        return img
        elif len(left_eye) == 0 or len(right_eye) == 0:
            left_eye = left_eye_cascade.detectMultiScale(img, 1.05, 3)
            right_eye = right_eye_cascade.detectMultiScale(img, 1.05, 3)
            if len(left_eye) > 0 and len(right_eye) > 0:
                (mag_x, mag_y) = mag_xy
                for (lex, ley, lew, leh) in left_eye:
                    for (rex, rey, rew, reh) in right_eye:

                        # adapting H W for each image
                        height, width = img.shape[:2]

                        # Magstripe position is to the right than both eyes
                        if mag_x > lex and mag_x > rex:
                            if ley > mag_y > rey:
                                above_eyes = round((lex + rex) / 2) - (leh + reh / 4)
                                beginX = 0
                                endX = above_eyes
                                beginY = 0
                                endY = height
                                top_img = img[beginY:endY, beginX:endX]
                                img_candidates.append(top_img)
                                img_orientation.append([beginY,endY, beginX,endX])
                        # Magstripe is between both eyes and is below it
                        elif lex < mag_x < rex:
                            if mag_y > lex and mag_y > rex:
                                above_eyes = round((ley + rey) / 2) - (leh + reh / 4)
                                beginX = 0
                                endX = width
                                beginY = 0
                                endY = above_eyes
                                top_img = img[beginY:endY, beginX:endX]
                                img_candidates.append(top_img)
                                img_orientation.append([beginY,endY, beginX,endX])
                        # Magstripe position is on the left side of both eyes
                        elif mag_x < lex and mag_x > rex:
                            if rey > mag_y > ley:
                                above_eyes = round((lex + rex) / 2) + (leh + reh / 4)
                                beginX = above_eyes
                                endX = width
                                beginY = 0
                                endY = height
                                top_img = img[beginY:endY, beginX:endX]
                                img_candidates.append(top_img)
                                img_orientation.append([beginY,endY, beginX,endX])
                        # Magstripe is between both eyes but is above it
                        elif lex < mag_x < rex:
                            if mag_y < ley and mag_y < rey:
                                above_eyes = round((ley + rey)/2) + (leh + reh / 4)
                                beginX = 0
                                endX = width
                                beginY = above_eyes
                                endY = height
                                top_img = img[beginY:endY, beginX:endX]
                                img_candidates.append(top_img)
                                img_orientation.append([beginY,endY, beginX,endX])
                        else:
                            print("could not crop image L3")
                            print((lex, ley, lew, leh), (rex, rey, rew, reh), (mag_x, mag_y))
                            return None
            elif len(left_eye) == 0 or len(right_eye) == 0:
                left_eye = left_eye_cascade.detectMultiScale(img, 1.05, 2)
                right_eye = right_eye_cascade.detectMultiScale(img, 1.05, 2)
                if len(left_eye) > 0 and len(right_eye) > 0:
                    (mag_x, mag_y) = mag_xy
                    for (lex, ley, lew, leh) in left_eye:
                        for (rex, rey, rew, reh) in right_eye:

                            # adapting H W for each image
                            height, width = img.shape[:2]

                            # Magstripe position is to the right than both eyes
                            if mag_x > lex and mag_x > rex:
                                if ley > mag_y > rey:
                                    above_eyes = round((lex + rex) / 2) - (leh + reh / 4)
                                    beginX = 0
                                    endX = above_eyes
                                    beginY = 0
                                    endY = height
                                    top_img = img[beginY:endY, beginX:endX]
                                    img_candidates.append(top_img)
                                    img_orientation.append([beginY,endY, beginX,endX])
                            # Magstripe is between both eyes and is below it
                            elif lex < mag_x < rex:
                                if mag_y > lex and mag_y > rex:
                                    above_eyes = round((ley + rey) / 2) - (leh + reh / 4)
                                    beginX = 0
                                    endX = width
                                    beginY = 0
                                    endY = above_eyes
                                    top_img = img[beginY:endY, beginX:endX]
                                    img_candidates.append(top_img)
                                    img_orientation.append([beginY,endY, beginX,endX])
                            # Magstripe position is on the left side of both eyes
                            elif mag_x < lex and mag_x > rex:
                                if rey > mag_y > ley:
                                    above_eyes = round((lex + rex) / 2) + (leh + reh / 4)
                                    beginX = above_eyes
                                    endX = width
                                    beginY = 0
                                    endY = height
                                    top_img = img[beginY:endY, beginX:endX]
                                    img_candidates.append(top_img)
                                    img_orientation.append([beginY,endY, beginX,endX])
                            # Magstripe is between both eyes but is above it
                            elif lex < mag_x < rex:
                                if mag_y < ley and mag_y < rey:
                                    above_eyes = round((ley + rey)/2) + (leh + reh / 4)
                                    beginX = 0
                                    endX = width
                                    beginY = above_eyes
                                    endY = height
                                    top_img = img[beginY:endY, beginX:endX]
                                    img_candidates.append(top_img)
                                    img_orientation.append([beginY,endY, beginX,endX])
                            else:
                                print("could not crop image L2")
                                print((lex, ley, lew, leh), (rex, rey, rew, reh), (mag_x, mag_y))
                                return None
    return img_candidates , img_orientation


def crop_above_eyes_w_recursion(img, mag_xy):
    if img is None:
        print("Image not loaded to crop")
        return None
    img_candidates = []

    img_candidates = above_eyes_recursive(img, mag_xy, img_candidates)

    return img_candidates

#modified crop above eyes function with recursion to allow for threshold sensitivity
def above_eyes_recursive(img, mag_xy, img_candidates, scaleFactor = 1.4, minNeighbors= 5):
    if img is None:
        print("Image not loaded to crop")
        return None
    # cascades for left amd right eye
    left_eye = left_eye_cascade.detectMultiScale(img, scaleFactor, minNeighbors)
    right_eye = right_eye_cascade.detectMultiScale(img, scaleFactor, minNeighbors)

    # adapting H W for each image in larger scope
    height, width = img.shape[:2]

    if len(left_eye)> 0 and len(right_eye) > 0 and scaleFactor >= 1.05:

        for (lex, ley, lew, leh) in left_eye:
            for (rex, rey, rew, reh) in right_eye:
                (mag_x, mag_y) = mag_xy
                # Magstripe position is to the right than both eyes
                if mag_x > lex and mag_x > rex:
                    if ley > mag_y > rey:
                        above_eyes = round(((lex + rex) / 2) - (leh + reh / 4))
                        beginX = 0
                        endX = above_eyes
                        beginY = 0
                        endY = height
                        top_img = img[beginY:endY, beginX:endX]
                        img_candidates.append(top_img)
                    return

                # Magstripe is between both eyes and is below it
                elif lex < mag_x < rex:
                    if mag_y > lex and mag_y > rex:
                        above_eyes = round(((ley + rey) / 2) - (leh + reh / 4))
                        beginX = 0
                        endX = width
                        beginY = 0
                        endY = above_eyes
                        top_img = img[beginY:endY, beginX:endX]
                        img_candidates.append(top_img)
                    return
                # Magstripe position is on the left side of both eyes
                elif mag_x < lex and mag_x > rex:
                    if rey > mag_y > ley:
                        above_eyes = round(((lex + rex) / 2) + (leh + reh / 4))
                        beginX = above_eyes
                        endX = width
                        beginY = 0
                        endY = height
                        top_img = img[beginY:endY, beginX:endX]
                        img_candidates.append(top_img)
                    return
                # Magstripe is between both eyes but is above it
                elif lex < mag_x < rex:
                    if mag_y < ley and mag_y < rey:
                        above_eyes = round(((ley + rey)/2) + (leh + reh / 4))
                        beginX = 0
                        endX = width
                        beginY = above_eyes
                        endY = height
                        top_img = img[beginY:endY, beginX:endX]
                        img_candidates.append(top_img)
                    return
                else:
                    end_else(img, mag_xy, img_candidates, scaleFactor, minNeighbors)

    else:
        end_else(img, mag_xy, img_candidates, scaleFactor, minNeighbors)

def end_else(img, mag_xy, img_candidates, scaleFactor, minNeighbors):
    print(scaleFactor, len(img_candidates), minNeighbors)
    if scaleFactor <= 1.071 and len(img_candidates) < 1 and minNeighbors >= 2:
        minNeighbors -= 1
        print ("minNeighbors: ", minNeighbors)
        img_candidates = above_eyes_recursive(img, mag_xy, img_candidates, scaleFactor, minNeighbors )
    elif len(img_candidates) >= 1:
        print("return images")
        return img_candidates
    elif 1.11 <= scaleFactor <= 1.4:
        scaleFactor -= 0.07
        print("scaleFactor: ", scaleFactor)
        img_candidates = above_eyes_recursive(img, mag_xy, img_candidates, scaleFactor, minNeighbors )
    else:
        print(scaleFactor, len(img_candidates), minNeighbors)
