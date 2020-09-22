import cv2
import os
import numpy as np

# protocol://host:port/script_name?script_params|auth
# Function splits incoming mp4 and returns


def split_frames(cap_origin_path):
    # Assigning video from file to variable cap
    cap = cv2.VideoCapture(cap_origin_path)
    # Starting with the first{0} frame

    images = []

    # CAP_PROP_POS_FRAMES = 1
    pos_frame = cap.get(1)

    while True:
        # (Retval, image) | Grabs, decodes, and returns for next frame
        retrieved, frame = cap.read()
        if retrieved:
            # Adds frame to list
            images.append(frame)

            # Updates frame count
            pos_frame = cap.get(1)
        else:
            cap.set(1, pos_frame)
            break

        # if frame position is the same as the expected frame count break loop
        if cap.get(1) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            break

    # Return value contains array of the individual frames
    split_frames_array = np.array(images)
    return split_frames_array
