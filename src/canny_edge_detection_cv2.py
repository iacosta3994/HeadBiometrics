import numpy as np
import cv2
from matplotlib import pyplot as plt

def make_canny(filename, filename_canny):
    img = cv2.imread(filename,0) #pulling img
    edges = cv2.Canny(img,100,200)              #making img canny
    cv2.imwrite(filename_canny, edges) #overwrite startingg frame with canny version


    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Canny Image'), plt.xticks([]), plt.yticks([])
    plt.show()

"""cap_origin_video, saved_frames_location,  frames_to_skip, current_frame_idx  = ('Video_Tests\A_Tilt_Side_to_Side.mp4'), ('./data/A_frame') , 30 , 0
filename = saved_frames_location + str(current_frame_idx) + '.jpeg'
filename_canny = saved_frames_location + '_canny'+ str(current_frame_idx) +  '.jpeg'

make_canny(filename, filename_canny)"""
