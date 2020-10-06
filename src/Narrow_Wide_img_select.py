import cv2
import numpy as np
from src.face_contour_width import *



def narrowest_img (img_array):     # Inputs frame-filename to scan for the narrowest head img
    #establishes both left and right eye
    left_eye_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
    right_eye_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')

    #establishing var outside loop
    dist == None
    ret_img = None

    #itteration for each img
    for img in img_array:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        left_eye = left_eye_cascade.detectMultiScale(gray, 1.3, 5)
        right_eye = right_eye_cascade.detectMultiScale(gray, 1.3, 5)
        #if both eyes found
        if left_eye and right_eye:
            #itterating through both combos of left and right
            for (lex, ley, lew, leh) in left_eye:
                for (rex, rey, rew, reh) in right_eye:
                    #distance between points of detected eyes
                    img_dist = np.sqrt((rex-lex)**2 + (rey -ley)**2)
                    #if dist not established first img creates value
                    if dist == None:
                        dist = img_dist
                    #when larger distance detected updates return img
                    elif img_dist > dist:
                        dist = img_dist
                        ret_img = img
                    #if the dist is not the largest or none continue to next img
                    else:
                        continue
        return ret_img

def widest_img(img_array):
    head_contour_list = []

    # itterating through each image in img array
    for img in img_array:
        # runs head contour that gets the contour from the img
        contour, img_canny = head_contour(img)

        # asses the contour using its area
        area = cv2.contourArea(contour)

        # if the list is empty asign the first one
        if not head_contour_list:
            head_contour_list.append([contour, area, img])
            continue
        # if the area of the img is larger than the past frames replace it
        if area > head_contour_list[0][1]:
            head_contour_list.pop(0)
            head_contour_list.append([contour, area, img])


        else:
            continue

    return head_contour_list[2]
