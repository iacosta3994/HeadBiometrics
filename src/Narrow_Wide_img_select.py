import cv2
import numpy as np
import types
from src.face_contour_width import head_contour

left_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')


def narrowest_img(img_array):     # Inputs frame-filename to scan for the narrowest head img

    # establishing var outside loop
    dist = None
    ret_img = None
    print(len(img_array))
    # itteration for each img
    for img in img_array:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        left_eye = left_eye_cascade.detectMultiScale(img, 1.05, 5)
        right_eye = right_eye_cascade.detectMultiScale(img, 1.05, 5)

        # if both eyes found
        if (type(left_eye) is tuple) and (type(right_eye) is tuple):
            print("success")
            # itterating through both combos of left and right
            for (lex, ley, lew, leh) in left_eye:
                for (rex, rey, rew, reh) in right_eye:
                    # distance between points of detected eyes
                    img_dist = np.sqrt((rex-lex)**2 + (rey - ley)**2)
                    # if dist not established first img creates value
                    if dist == None:
                        dist = img_dist
                    # when larger distance detected updates return img
                    if img_dist > dist:
                        dist = img_dist
                        ret_img = img
                    # if the dist is not the largest or none continue to next img
                    else:
                        continue



    return ret_img



def widest_img(img_array, img_orientation):

    widest_head_img = None
    largest_contour = None
    # itterating through each image in img array
    for img in img_array:
        # runs head contour that gets the contour from the img
        for orientation in img_orientation:
            beginY, endY, beginX, endX = orientation
            og_img = img.copy()
            img = img[beginY:endY, beginX:endX]
            contour, img_canny = head_contour(img)

            contour_length = cv2.arcLength(contour, closed=False)
            if largest_contour == None or contour_length > largest_contour:
                cv2.drawContours(img, contour, -1, (0,255,0), 3)
                cv2.imshow('Widest img', img_canny)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                largest_contour = contour_length
                widest_head_img = og_img
            else:
                continue

    return widest_head_img
