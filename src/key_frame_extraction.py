import cv2
import numpy as np
import os

global final_frame_idx                                                          # Global variable, split frames will add an int indicating which frame it worked on
def split_frames(cap_origin_path, saved_frames_location, frames_to_skip):       # Function splits incoming mp4 and sends finished frames to saved_frames_locations
    if not os.path.exists('data'):                                          # This checks if directory is made
            os.makedirs('data')                                                 # This makes the directory to store the images if no file was found
    cap = cv2.VideoCapture(cap_origin_path)                                     # Assigning video from file to variable cap
    current_frame_idx = 0                                                       # Starting with the first{0} frame
    while cap.isOpened():                                                       # While there are still images in cap: do the following
        filename = saved_frames_location + str(current_frame_idx) + '.jpeg'     #   Establishing filename with the sequential number of frame
        retrieved, frame = cap.read()                                           #   (Retval, image) | Grabs, decodes, and returns for next frame
        if retrieved:                                                           #   True if an image was retrieved
            cv2.imwrite(filename, frame)                                        #   Save the file with filename, and the frame in while loop
            final_frame_idx = current_frame_idx                                 #   Assigning final_frame_idx to current_frame_idx current is changed for next itteration
            current_frame_idx += frames_to_skip                                 #   Prepare current for next itteration
            cap.set(1, current_frame_idx)                                       #   Set(propId, value)| Allows to skip to later frame in cap, new loop occurs
        else:                                                                   #       If no image was inputed
            cap.release()                                                       #       Deallocates used memory and clears capture pointer
            return(final_frame_idx)                                             #       Return value will be used to determine stopping frame in A&B_quantify, Loop ends
