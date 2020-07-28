import numpy as np
import cv2
from matplotlib import pyplot as plt

def make_canny(filename, filename_canny):
    img = cv2.imread(filename,0)   #pulling img
    edges = cv2.Canny(img,100,200) #making img canny
    cv2.imwrite(filename_canny, edges) #writes frame with canny version

    #plt.subplot(121),plt.imshow(img,cmap = 'gray')
    #plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    #plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    #plt.title('Canny Image'), plt.xticks([]), plt.yticks([])
    #plt.show()
