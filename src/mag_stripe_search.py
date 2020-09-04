
import os
import sys
import numpy as np
import matplotlib
import cv2
import uuid

# End function that locates the contours in search for a rectangle with deminsions similar to a magnetic stripe
def get_magstripe_demensions(img):
    # Generates a hierarchy of contours
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Detects contours with 4 sides and an aspec ratio relating to magstripe
    mag_stripe_h_w = detect_mag_stripe(img, contours)
    # Returns the pixel height and width with angle in mind
    return mag_stripe_h_w


# Narrows down the contours in search for the one that best fits
def detect_mag_stripe(img, contours):
    magstripe_list = []
    for contour in contours:
        # Epsilon helps account for possible inperfections in shape of magstripe (Glare, off angles, or from the Blurring)
        epsilon = (0.04 * cv2.arcLength(contour, True))
        # Apply contour approximation and store the result in approx
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # If the contour has 4 sides
        if len(approx) == 4:
            rect = cv2.minAreaRect(contour)
            # Creates a variable with the rectangles data
            (x, y), (width, height), angle = rect
            # Rectangle can be flipped because image in portrait mode is in landscape and can be one side or the other
            aspectRatio = min(width, height) / max(width, height)
            # Has aspect ratio to look for card in diffrent angles. .1 is true demensions, but some visual distortion exsists
            if aspectRatio >= .07 and aspectRatio <= .16:
                # If the box is bigger than 10 by 10 pixels, to filter out artifacts
                if height > 5 and width > 5:
                    #print (aspectRatio )
                    # The pixel height and width of card is returned
                    magstripe_name = str(uuid.uuid4())
                    magstripe_list.append([width, height, width * height, magstripe_name])
                    # Test drawing function
                    test_draw_magstripe_contour(img, contour, magstripe_name)
    return magstripe_list



# Test function that draws a bounding box around contours that detects as magstripe
def test_draw_magstripe_contour(img, contour, magstripe_name, color=(0, 255, 0)):
    #easier to reestablish values than to port over
    rect = cv2.minAreaRect(contour)
    (x, y), (width, height), angle = rect
    #Finds the center of each contour to label
    M = cv2.moments(c)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])

    # Using the contour box points are created
    box = cv2.boxPoints(rect)
    # Int64 using boxPoints as input
    box = np.int0(box)

    cv2.putText(img, text= str(magstripe_name), org=(cx,cy),
            fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0),
            thickness=2, lineType=cv2.LINE_AA)
    # Creates green box to output file path
    cv2.imwrite('img' + magstripe_name + '.png', cv2.drawContours(img, [box], 0, color, 2))
