
import os
import sys
import numpy as np
import matplotlib
import cv2



def get_magstripe_demensions(canny_image, filename_canny):

    contours, hierarchy = cv2.findContours(canny_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mag_stripe_h_w = detect_mag_stripe(contours, filename_canny)
    return mag_stripe_h_w

def test_draw_magstripe_contour(contour, filename_canny, color = (0,255,0)):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    canny_image_write = cv2.imread(filename_canny)
    cv2.imwrite(filename_canny, cv2.drawContours(canny_image_write,[box],0,color,2))

def detect_mag_stripe(contours, filename_canny):
    for contour in contours:
        # calculate epsilon  using
        epsilon = (0.04 * cv2.arcLength(contour, True))
        # apply contour approximation and store the result in approx
        approx = cv2.approxPolyDP(contour, epsilon , True)

        if len(approx ) == 4:
            rect = cv2.minAreaRect(contour)
            (x, y), (width, height), angle = rect
            aspectRatio = min(width, height) / max(width, height)
            if aspectRatio >= .075 and aspectRatio <= .16:
                if height > 10 and width > 10:
                    test_draw_magstripe_contour(contour,filename_canny)
                    return (width, height)
