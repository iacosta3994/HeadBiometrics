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

mag_stripe_constant_w_h = [85.60, 8.37]

def get_magstripe_demensions (canny_image, filename_canny):

    contours, hierarchy = cv2.findContours(canny_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mag_stripe_w_h = detect_mag_stripe(contours, filename_canny)
    return mag_stripe_w_h

def test_draw_magstripe_contour(contour, filename_canny):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    canny_image_write = cv2.imread(filename_canny)
    cv2.imwrite(filename_canny, cv2.drawContours(canny_image_write,[box],0,(0,0,255),2))

def detect_mag_stripe(contours, filename_canny):
    for contour in contours:
        # calculate epsilon  using
        epsilon  = (0.05 * cv2.arcLength(contour, True))
        # apply contour approximation and store the result in approx
        approx  = cv2.approxPolyDP(contour, 0.05 * epsilon , True)

        if len(approx ) == 4:
            x, y, width, height = cv2.boundingRect(approx )
            aspectRatio = float(width) / height
            if aspectRatio >= 9.5 and aspectRatio <= 12.5:
                test_draw_magstripe_contour(contour, filename_canny)
                return width, height

#next step will be to get the largest boxpoints L35 to return the most accurate pixel to mm measurement.
