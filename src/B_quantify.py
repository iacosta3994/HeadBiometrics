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


cap_origin_video, saved_frames_location,  frames_to_skip  = ('Video_Tests\B_Turn_Left_Right.mp4'), ('./data/B_frame') , 10

final_frame_idx = split_frames(cap_origin_video, saved_frames_location, frames_to_skip)
current_frame_idx = 0

while os.path:

    filename, filename_canny = saved_frames_location + str(current_frame_idx) + '.jpeg' , saved_frames_location + '_canny'+ str(current_frame_idx) +  '.jpeg'

    with open('filename', 'w') as f:
        if current_frame_idx <= final_frame_idx:
            canny_image = make_canny(filename, filename_canny)
            get_magstripe_demensions(canny_image, filename_canny)
            #print("Current frame: {} Final frame : {}".format(current_frame_idx, final_frame_idx))
            current_frame_idx += frames_to_skip
        else:
            break
