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

def video_to_pixel_mm (origin_video, saved_frame_name, use_every_x_frame):                                              # Funtion that inputs video mp4, name of scene with location, variable to determine how many frames to splice

    cap_origin_video, saved_frames_location,  frames_to_skip  = (origin_video, saved_frame_name , use_every_x_frame)    # Assinging the input variables to local variables for Function

    final_frame_idx = split_frames(cap_origin_video, saved_frames_location, frames_to_skip)                             # Splits the frame then returns the frame that is being worked on
    current_frame_idx = 0                                                                                               # Starting with the first frame

    list_mag_stripe_w_h = []                                                                                             # Later this list will have all the H and W of the mag stripe
    mag_stripe_constant_w_h = [85.60, 8.37]                                                                             # Measured Magstripe is 85.60 mm wide and 8.37 mm tall


    while os.path:

        filename = saved_frames_location + str(current_frame_idx) + '.jpeg'                                             # The name of the frame being proccesed
        filename_canny = saved_frames_location + '_canny'+ str(current_frame_idx) +  '.jpeg'                            # Similar to prior but indicates the version with canny that is used for printing out the test

        with open('filename', 'w') as f:                                                                                # While there are frames to process continue
            if current_frame_idx <= final_frame_idx:                                                                    # If the current frame is still part of the video || continue
                canny_image = make_canny_magstripe(filename, filename_canny)                                            # Variable now has the canny image to process
                mag_stripe_w_h = get_magstripe_demensions(canny_image, filename_canny)                                  # canny_image var is then used with get_magstripe_demensions
                if mag_stripe_w_h:                                                                                      # It checks if any width and height was in the var
                    list_mag_stripe_w_h.append(mag_stripe_w_h)                                                          # Appends data into list mag stripe to be consolidated after it finishes with loop
                current_frame_idx += frames_to_skip                                                                     # Ticks the counter for the next idx

            else:                                                                                                       # Once all the frames have been processed continue below
                if len(list_mag_stripe_w_h) == 0:                                                                       # If list_mag_stripe_w_h didnt return anything stop the loop
                    print("found no magstripe")
                    break

                #if ((len(list_mag_stripe_w_h)) > 11):                                                                   # Checks if lsit of data is longer than 10 recorded numbers
                #    list_mag_stripe_w_h = sorted(list_mag_stripe_w_h)                                                   # Sorts the list
                #    list_mag_stripe_w_h = list_mag_stripe_w_h[10:]                                                       # Removes the smallest var in list

                '''
                standard deviation to remove the outliers in list_mag_stripe_w_h using [2] as factor to remove from list or not

                '''


                mean_mag_stripe_w_h = (np.mean(list_mag_stripe_w_h, axis=0))                                            # Creates an avg of the W and H variables

                pixel_mm_w_h = (np.divide(mag_stripe_constant_w_h, mean_mag_stripe_w_h))                                # This divides measured H and W witht the avg this will have a H and W square

                pixel_mm_mean = ((pixel_mm_w_h[0] + pixel_mm_w_h[1])/2)                                                 # Removes the variable from a square to an avg unit of h and w

                return pixel_mm_mean
#video_to_pixel_mm(('Video_Tests\B_test_Tom.mp4'), ('./data/B_frame') , 1)
