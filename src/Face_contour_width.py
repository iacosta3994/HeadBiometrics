import os
import cv2
import uuid
import numpy as np
from src.canny_edge_detection_cv2 import neural_edge_detection


def img_head_contour_array(img_samples_array):
    main_canny = None
    main_contour = None
    main_contour_length = None

    if len(img_samples_array) == 0:
        print("Can't open sample array")
        return main_canny, main_contour, main_contour_length

    for img in img_samples_array:

        img_duplicate = img.copy()

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        contour, img_canny = neural_edge_detection(img, ret_contour=True)

        if contour is None:
            continue

        cv2.drawContours(img_duplicate, contour, -1, (0,255,0), 3)
        cv2.imshow('Contours', img_canny)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        contour_length = cv2.arcLength(contour, closed=False)


        if contour_length is None:
            print("contour length is none on face contour width")
            continue

        if main_contour_length is None or contour_length > main_contour_length:
            main_canny = img_canny
            main_contour = contour
            main_contour_length = contour_length
        else:
            continue

    if main_canny is None and main_contour is None and main_contour_length is None:
        print("no return values found")
    return  main_contour, main_contour_length


# outputs the contour of the head

def img_head_contour(img):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    contour, img_canny = neural_edge_detection(img, ret_contour=True)
    img_duplicate = img_canny.copy()
    cv2.drawContours(img_duplicate, contour, -1, (0,255,0), 3)
    cv2.imshow('head contour', img_canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    contour_length = cv2.arcLength(contour, closed=False)

    return  contour, contour_length
