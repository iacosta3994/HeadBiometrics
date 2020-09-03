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
from src.key_frame_extraction import *
from src.canny_edge_detection_cv2 import *
from src.mag_stripe_search import *


# Funtion that inputs video mp4, name of scene with location, variable to determine how many frames to splice
def video_to_pixel_mm(origin_video, saved_frame_name, use_every_x_frame):

    # Assinging the input variables to local variables for Function
    cap_origin_video, saved_frames_location,  frames_to_skip = (
        origin_video, saved_frame_name, use_every_x_frame)

    # Splits the frame then returns the frame that is being worked on
    final_frame_idx = split_frames(cap_origin_video, saved_frames_location, frames_to_skip)
    # Starting with the first frame
    current_frame_idx = 0

    # Later this list will have all the H and W of the mag stripe
    list_mag_stripe_w_h = []
    # Measured Magstripe is 85.60 mm wide and 8.37 mm tall
    mag_stripe_constant_w_h = [85.60, 8.37]

    while os.path:

        # The name of the frame being proccesed
        filename = saved_frames_location + str(current_frame_idx) + '.jpeg'
        # Similar to prior but indicates the version with canny that is used for printing out the test
        filename_canny = saved_frames_location + '_canny' + str(current_frame_idx) + '.jpeg'

        # While there are frames to process continue
        with open('filename', 'w') as f:
            # If the current frame is still part of the video || continue
            if current_frame_idx <= final_frame_idx:
                # Variable now has the canny image to process
                canny_image = make_canny_magstripe(filename, filename_canny)
                # canny_image var is then used with get_magstripe_demensions
                mag_stripe_w_h = get_magstripe_demensions(canny_image, filename_canny)
                # It checks if any width and height was in the var
                if mag_stripe_w_h:
                    # Appends data into list mag stripe to be consolidated after it finishes with loop
                    list_mag_stripe_w_h.append(mag_stripe_w_h)
                # Ticks the counter for the next idx
                current_frame_idx += frames_to_skip

            # Once all the frames have been processed continue below
            else:
                # If list_mag_stripe_w_h didnt return anything stop the loop
                if len(list_mag_stripe_w_h) == 0:
                    print("found no magstripe")
                    break
                with open('magstripe_pixel_w_h_a.txt', 'w') as file_txt:
                    for w_h_a in list_mag_stripe_w_h:
                        for element in w_h_a:
                            file_txt.write(str(element) + " ")
                        file_txt.write('\n')
                # if ((len(list_mag_stripe_w_h)) > 11):                                                                   # Checks if lsit of data is longer than 10 recorded numbers
                #    list_mag_stripe_w_h = sorted(list_mag_stripe_w_h)                                                   # Sorts the list
                #    list_mag_stripe_w_h = list_mag_stripe_w_h[10:]                                                       # Removes the smallest var in list

                '''
                standard deviation to remove the outliers in list_mag_stripe_w_h using [2] as factor to remove from list or not

                '''

                # Creates an avg of the W and H variables
                mean_mag_stripe_w_h = (np.mean(list_mag_stripe_w_h, axis=0))

                # pixel_mm_w_h = (np.divide(mag_stripe_constant_w_h, mean_mag_stripe_w_h))                                # This divides measured H and W witht the avg this will have a H and W square

                # pixel_mm_mean = ((pixel_mm_w_h[0] + pixel_mm_w_h[1])/2)                                                 # Removes the variable from a square to an avg unit of h and w
                pixel_mm_mean = True
                return pixel_mm_mean


video_to_pixel_mm(('Video_Tests\B_test_self.mp4'), ('./data/B_frame'), 1)
