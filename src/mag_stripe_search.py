
import os
import sys
import numpy as np
import cv2
import uuid

# End function that locates the contours in search for a rectangle with deminsions similar to a magnetic stripe


def get_magstripe_demensions(img):
    # copies image
    img_fc = img.copy()
    # Generates a hierarchy of contours
    contours, hierarchy = cv2.findContours(img_fc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Detects contours with 4 sides and an aspec ratio relating to magstripe
    mag_stripe_w_h_ar_mn_xy = detect_mag_stripe(img, contours)
    # Returns the pixel height and width with angle in mind
    return mag_stripe_w_h_ar_mn_xy


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
            if width < height:
                width, height = height, width
            if height <= 15:
                continue
            aspectRatio = height / width
            if 0.09288888888 <= aspectRatio <= 0.1653734246:
                # The pixel height and width of card is returned
                magstripe_name = str(uuid.uuid4())
                magstripe_list.append([width, height, width * height,
                                       aspectRatio, magstripe_name, (x, y)])
                # Test drawing function
                #test_draw_magstripe_contour(img, contour, magstripe_name)
    return magstripe_list


# Test function that draws a bounding box around contours that detects as magstripe
def test_draw_magstripe_contour(img, contour, magstripe_name, color=(0, 255, 0)):
    # easier to reestablish values than to port over
    rect = cv2.minAreaRect(contour)
    (x, y), (width, height), angle = rect
    # Finds the center of each contour to label

    # Using the contour box points are created
    box = cv2.boxPoints(rect)
    # Int64 using boxPoints as input
    box = np.int0(box)

    M = cv2.moments(contour)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    folder_check = os.path.isdir("Data")
    if not folder_check:
        os.makedirs("Data")
    cv2.putText(img, text=str(magstripe_name), org=(cx, cy),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0),
                thickness=2, lineType=cv2.LINE_AA)
    # Creates green box to output file path
    cv2.imwrite('Data/'+'img_' + magstripe_name + '.png', cv2.drawContours(img, [box], 0, color, 2))
