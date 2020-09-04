
import os
import sys
import numpy as np
import matplotlib
import cv2
import uuid

# End function that locates the contours in search for a rectangle with deminsions similar to a magnetic stripe
def get_magstripe_demensions(canny_image):
    # Generates a hierarchy of contours
    contours, hierarchy = cv2.findContours(canny_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Detects contours with 4 sides and an aspec ratio relating to magstripe
    mag_stripe_h_w = detect_mag_stripe(canny_image, contours)
    # Returns the pixel height and width with angle in mind
    return mag_stripe_h_w


# Test function that draws a bounding box around contours that detects as magstripe
def test_draw_magstripe_contour(img, contour, color=(0, 255, 0)):
    # Incoming contour is inputed into var rect
    rect = cv2.minAreaRect(contour)
    # Using the contour box points are created
    box = cv2.boxPoints(rect)
    # Int64 using boxPoints as input
    box = np.int0(box)
    # Opens the canny image that will be written on

    # Creates green box to output file path
    cv2.imwrite('filename_canny.png', cv2.drawContours(img, [box], 0, color, 2))


# Narrows down the contours in search for the one that best fits
def detect_mag_stripe(canny_image, contours):
    for contour in contours:
        # Epsilon helps account for possible inperfections in shape of magstripe (Glare, off angles, or from the Blurring)
        epsilon = (0.04 * cv2.arcLength(contour, True))
        # Apply contour approximation and store the result in approx
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # If the contour has 4 sides
        if len(approx) == 4:
            # Creates a variable with the rectangles data
            (x, y), (width, height), angle = cv2.minAreaRect(contour)
            # Rectangle can be flipped because image in portrait mode is in landscape and can be one side or the other
            aspectRatio = min(width, height) / max(width, height)
            # Has aspect ratio to look for card in diffrent angles. .1 is true demensions, but some visual distortion exsists
            if aspectRatio >= .07 and aspectRatio <= .16:
                # If the box is bigger than 10 by 10 pixels, to filter out artifacts
                if height > 5 and width > 5:
                    # Test drawing function
                    #test_draw_magstripe_contour(canny_image, contour)
                    #print (aspectRatio )
                    # The pixel height and width of card is returned
                    magstripe_name = str(uuid.uuid4())
                    return [width, height, width * height, magstripe_name]
