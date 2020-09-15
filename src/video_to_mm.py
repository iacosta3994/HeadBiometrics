'''
This will be in charge of preparing B_photos (the side of the head). Then it will need to determine the front_to_back, length, circumference and ear_to_ear in pixels.
User input will be needed to find front_to_back due to dynamic hairlines in individuals heads.
then use info from mag_stripe_search to get the conversion from pixel to mm^2 giving front_to_back, length, circumference and ear_to_ear in mm
'''
import os
import sys
import numpy as np
import matplotlib
import cv2
from operator import truediv
from src.key_frame_extraction import *
from src.canny_edge_detection_cv2 import *
from src.mag_stripe_search import *
from src.face_detect_auto_crop import *
from src.img_demensions import *

# Funtion that inputs video mp4, name of scene with location, variable to determine how many frames to splice


def video_to_pixel_mm(cap_origin_path):

    # Splits the frame then returns the array of frames
    split_frame_array = split_frames(cap_origin_path)

    frame_proccessed = 0

    # Later this list will have all the H and W of the mag stripe
    list_mag_stripe_w_h_area_ratio_magname = []
    # Measured Magstripe 1 is 85.60 mm wide and 8.37 mm tall
    mag_stripe_constant_w_h_1 = [85.725, 8.382]
    mag_stripe_constant_w_h_2 = [85.725, 13.50155887997605]
    while (len(split_frame_array)) >= frame_proccessed:
        for frame in split_frame_array:

            # face detect auto crop input frame output face crop

            #face_crop = face_detect_auto_crop(frame, False)
            # if np.any(face_crop) != None:

            # Canny function for magstripe benchmark
            canny_image = make_canny_magstripe(frame)

            # canny_image var is then used with get_magstripe_demensions
            frame_mag_stripe_w_h_ar_mn_list = get_magstripe_demensions(canny_image)

            for mag_stripe_w_h_area_ratio_magname in frame_mag_stripe_w_h_ar_mn_list:
                # It checks if any width and height was in the var
                if mag_stripe_w_h_area_ratio_magname:
                    # Appends data into list mag stripe to be consolidated after it finishes with loop
                    list_mag_stripe_w_h_area_ratio_magname.append(mag_stripe_w_h_area_ratio_magname)
                    # Ticks the counter for the next frame
            frame_proccessed += 1

            # Once all the frames have been processed continue below
    '''
    with open('magstripe_pixel_w_h_a_n.txt', 'w') as file_txt:
        for w_h_ar_n in list_mag_stripe_w_h_area_ratio_magname:
            file_txt.write(str(w_h_ar_n) + ' ' + '\n')
    '''



    mean_mag_stripe_area = [np.mean(list_mag_stripe_w_h_area_ratio_magname[0], axis=0)]


    '''
    standard deviation to remove the outliers in list_mag_stripe_w_h_area_ratio_magname using [2] as factor to remove from list or not
    make sure its width height and not swapped
    '''

    mag_stripe_w_h = []
    for mag_stripe_w_h_area_ratio_magname in list_mag_stripe_w_h_area_ratio_magname:
        if 0.09288888888 <= mag_stripe_w_h_area_ratio_magname[2] <= 0.10266666666:
            mag_stripe_w_h = mag_stripe_w_h_area_ratio_magname[:1]
            # Creates an avg of the W and H variables
            mean_mag_stripe_w_h = (np.mean(mag_stripe_w_h, axis=0))
            # This divides measured H and W witht the avg this will have a H and W square
            pixel_mm_w_h = (np.divide(mag_stripe_constant_w_h_1, mean_mag_stripe_w_h))
            # Removes the variable from a square to an avg unit of h and w
            pixel_mm_mean = ((pixel_mm_w_h[0] + pixel_mm_w_h[1])/2)
            return pixel_mm_mean

        elif 0.14962357464 <= mag_stripe_w_h_area_ratio_magname[2] <= 0.1653734246:
            mag_stripe_w_h = mag_stripe_w_h_area_ratio_magname[:1]
            # Creates an avg of the W and H variables
            mean_mag_stripe_w_h = [np.mean(mag_stripe_w_h, axis=0)]
            # This divides measured H and W witht the avg this will have a H and W square
            pixel_mm_w_h = (np.divide(mag_stripe_constant_w_h_2, mean_mag_stripe_w_h))
            # Removes the variable from a square to an avg unit of h and w
            pixel_mm_mean = ((pixel_mm_w_h[0] + pixel_mm_w_h[1])/2)
            return pixel_mm_mean

        else:
            return ("error magstripe detected not within ratio bounds")

video_to_pixel_mm('Video_Tests\A_test_Tom.mp4')
