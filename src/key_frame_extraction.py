import cv2
import os
import array as arr


# Function splits incoming mp4 and sends finished frames to saved_frames_locations
def split_frames(cap_origin_path, video_name, frames_to_skip):
    # Assigning video from file to variable cap
    cap = cv2.VideoCapture(cap_origin_path)
    retrieved, frame = cap.read()  # (Retval, image) | Grabs, decodes, and returns for next frame
    # Starting with the first{0} frame
    current_frame_idx = 0
    split_frame_array = arr.array()
    while cap.isOpened():                                                       # While there are still images in cap: do the following
        # Establishing filename with the sequential number of frame
        filename = video_name + str(current_frame_idx) + '.jpeg'

        if retrieved:  # True if an image was retrieved
            # cv2.imwrite(filename, frame)                                        #   Save the file with filename, and the frame in while loop
            current_frame_idx += frames_to_skip  # Prepare current for next itteration
            # Set(propId, value)| Allows to skip to later frame in cap, new loop occurs
            cap.set(1, current_frame_idx)
            split_frame_array.append(filename, frame)
        else:  # If no image was inputed
            cap.release()  # Deallocates used memory and clears capture pointer
            # Return value will be used to determine stopping frame in A&B_quantify, Loop ends
            return(current_frame_idx, split_frame_array)
