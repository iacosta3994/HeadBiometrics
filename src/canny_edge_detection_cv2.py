import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def make_canny(starting_frame_location):
#starting off with the first fram in the database [frame0]
    img = cv.imread(starting_frame_location,0)
    edges = cv.Canny(img,100,200)

    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()
