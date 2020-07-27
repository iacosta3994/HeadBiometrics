'''
This will be in charge of preparing A_photos (front of the face) to look for the pixel count for breadth and ear_to_ear
Then It will need to determine the pixel breadth from side to side, along with the pixel length circumference over the top of the head.
and then use info from mag_stripe_search to get the conversion from pixel to mm^2 giving breadth and ear_to_ear in mm
'''
import os
import sys
import numpy as np
import matplotlib
import cv2
from src.key_frame_extraction import *
from src.canny_edge_detection_cv2 import *

cap_origin_video, saved_frames_location,  frames_to_skip  = ('Video_Tests\A_Tilt_Side_to_Side.mp4'), ('./data/A_frame') , 30

final_frame_idx = split_frames(cap_origin_video, saved_frames_location, frames_to_skip)

current_frame_idx = 0

while os.path:
    filename = saved_frames_location + str(current_frame_idx) + '.jpeg'
    filename_canny = saved_frames_location + 'canny'+ str(current_frame_idx) +  '.jpeg'
    with open('filename', 'w') as f:
        if current_frame_idx <= final_frame_idx:
            make_canny(filename,filename_canny)
            print("Current frame: {} Final frame : {}".format(current_frame_idx, final_frame_idx))
            current_frame_idx += frames_to_skip
        else:
            break
