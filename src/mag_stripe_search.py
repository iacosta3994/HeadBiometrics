
import os
import sys
import numpy as np
import matplotlib
import cv2



def get_magstripe_demensions(canny_image, filename_canny):                                          # End function that locates the contours in search for a rectangle with deminsions similar to a magnetic stripe
    contours, hierarchy = cv2.findContours(canny_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)     # Generates a hierarchy of contours
    mag_stripe_h_w = detect_mag_stripe(contours, filename_canny)                                    # Detects contours with 4 sides and an aspec ratio relating to magstripe
    return mag_stripe_h_w                                                                           # Returns the pixel height and width with angle in mind

def test_draw_magstripe_contour(contour, filename_canny, color = (0,255,0)):                        # Test function that draws a bounding box around contours that detects as magstripe
    rect = cv2.minAreaRect(contour)                                                                 # Incoming contour is inputed into var rect
    box = cv2.boxPoints(rect)                                                                       # Using the contour box points are created
    box = np.int0(box)                                                                              # Int64 using boxPoints as input
    canny_image_write = cv2.imread(filename_canny)                                                  # Opens the canny image that will be written on
    cv2.imwrite(filename_canny, cv2.drawContours(canny_image_write,[box],0,color,2))                # Creates green box to output file path

def detect_mag_stripe(contours, filename_canny):                                                    # Narrows down the contours in search for the one that best fits
    for contour in contours:
        epsilon = (0.03 * cv2.arcLength(contour, True))                                             # Calculate epsilon  using
        approx = cv2.approxPolyDP(contour, epsilon , True)                                          # Apply contour approximation and store the result in approx

        if len(approx ) == 4:                                                                       # If the contour has 4 sides
            rect = cv2.minAreaRect(contour)                                                         # Creates a variable with the rectangles data
            (x, y), (width, height), angle = rect                                                   # Assignes values from rect to get the height and width and location of the rectangle
            aspectRatio = min(width, height) / max(width, height)                                   # Rectangle can be flipped so did min max  to account for this issue
            #test_draw_magstripe_contour(contour, filename_canny)                                   # Test drawing function
            if aspectRatio >= .075 and aspectRatio <= .16:                                          # Has aspect ratio to look for card in diffrent angles. .1 is true demensions, but some visual distortion exsists
                if height > 10 and width > 10:                                                      # If the box is bigger than 10 by 10 pixels, to filter out artifacts
                    return (width, height)                                                          # The pixel height and width of card is returned
