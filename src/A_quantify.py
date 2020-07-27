'''
This will be in charge of preparing A_photos (front of the face) to look for the pixel count for breadth and ear_to_ear
Then It will need to determine the pixel breadth from side to side, along with the pixel length circumference over the top of the head.
and then use info from mag_stripe_search to get the conversion from pixel to mm^2 giving breadth and ear_to_ear in mm
'''
import os
import sys
from key_frame_extraction import split_frames
from canny_edge_detection_cv2 import make_canny

cap_origin_video, saved_frames_location,  frames_to_skip  = ('Video_Tests\A_Tilt_Side_to_Side.mp4'), ('./data/frame') , 10

split_frames(cap_origin_video, saved_frames_location, frames_to_skip)


current_frame_idx = 0
filename = saved_frames_location + str(current_frame_idx) + '.jpeg'

with open('filename', 'rw') as f:
    make_canny(filename)
    current_frame_idx += frames_to_skip
