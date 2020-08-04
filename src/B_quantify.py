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


cap_origin_video, saved_frames_location,  frames_to_skip  = ('Video_Tests\B_test_self.mp4'), ('./data/B_frame') , 1

final_frame_idx = split_frames(cap_origin_video, saved_frames_location, frames_to_skip)
current_frame_idx = 0

list_mag_stripe_w_h =[]
mag_stripe_constant_w_h = [85.60, 8.37]


while os.path:

    filename, filename_canny = saved_frames_location + str(current_frame_idx) + '.jpeg' , saved_frames_location + '_canny'+ str(current_frame_idx) +  '.jpeg'

    with open('filename', 'w') as f:
        if current_frame_idx <= final_frame_idx:
            canny_image = make_canny(filename, filename_canny)
            mag_stripe_w_h = get_magstripe_demensions(canny_image, filename_canny)
            if mag_stripe_w_h:
                list_mag_stripe_w_h.append(mag_stripe_w_h)
            #print("Current frame: {} Final frame : {}".format(current_frame_idx, final_frame_idx))
            current_frame_idx += frames_to_skip
        else:
            if len(list_mag_stripe_w_h) == 0:
                print("found no magstripe")
                break
            mean_mag_stripe_w_h = (np.mean(list_mag_stripe_w_h, axis=0))
            pixel_mm_w_h = (np.divide(mag_stripe_constant_w_h, mean_mag_stripe_w_h))
            pixel_mm_mean = ((pixel_mm_w_h[0] + pixel_mm_w_h[1])/2)
            print(pixel_mm_mean)
