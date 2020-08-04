import numpy as np
import cv2
from matplotlib import pyplot as plt

def make_canny(filename, filename_canny):         # Function will load frame from file_path and will save canny edge image in same directory with new name
    image = cv2.imread(filename,0)                       # Assigning variable img from frame in file_path
    blurred_img = cv2.GaussianBlur(image, (3,3), 0)     # Blurring to reduce artifacts
    img_canny = auto_canny(blurred_img)                  # Create edges around  image with auto min and max values
    cv2.imwrite(filename_canny, img_canny)             # Writes frame with canny name to file path
    return img_canny
def auto_canny(img, sigma=0.33):                    # Runs an automated version of canny with optimal lower and upper thresholds || sigma .33 Default
    v = np.median(img)                                  # Gets median value of channel intensity | unique for each image
    lower = int(max(0, (1.0 - sigma) * v))              # Sets lower int valeue for canny function
    upper = int(min(255, (1.0 + sigma)* v))             # Sets higher int value for canny function
    img_canny = cv2.Canny(img , lower , upper)             # Img_canny now has the canny edge image
    return img_canny


    #plt.subplot(121),plt.imshow(img,cmap = 'gray')
    #plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    #plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    #plt.title('Canny Image'), plt.xticks([]), plt.yticks([])
    #plt.show()
