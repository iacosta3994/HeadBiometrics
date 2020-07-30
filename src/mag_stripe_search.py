'''
Inputs will be a database of images, that have already been modified with canny.
Function will look for a set of pixels with same ratio demensions as mag_stripe.

With the image, Height and Width (pixel_H, pixel_W : 0 , 0)will be saved along with specific frameIdx it belongs to
This will help determine wich frame has the Highest Height & Width, new py that esablishes collection of values to find Highest values will be output
'''
import os
import sys
import numpy as np
import matplotlib
import cv2



def get_magstripe_demensions (canny_image, filename_canny):

    contours, hierarchy = cv2.findContours(canny_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    detect_mag_stripe(contours, filename_canny)


def detect_mag_stripe(contours, filename_canny):
    for contour in contours:
        # calculate perimeter using
        perimeter = cv2.arcLength(contour, True)
        # apply contour approximation and store the result in vertices
        vertices = cv2.approxPolyDP(contour, 0.05 * perimeter, True)

        if len(vertices) == 3:
            x, y, width, height = cv2.boundingRect(vertices)
            aspectRatio = float(width) / height
            print(aspectRatio)
            if aspectRatio >= .5 and aspectRatio <= 10.5:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                canny_image_write = cv2.imread(filename_canny)
                cv2.imwrite(filename_canny, cv2.drawContours(canny_image_write,[box],0,(0,0,255),2))
