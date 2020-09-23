import numpy as np
import cv2

# Function will load frame from file_path and will save canny edge image in same directory with new name
def make_canny_magstripe(img):

    gray_img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY)
    # Blurring to reduce artifacts
    blurred_img = cv2.GaussianBlur(gray_img, (7, 7), 0)
    clahe = cv2.createCLAHE()  # Contrast Limited Adaptive Histogram Equalization
    # Adjsuts contrast in image to allow maggstripe to be darker for images with high gamma
    clahe_img = clahe.apply(blurred_img)
    # 35 works well Sets a threshold in search for the pixels near the color of black i.e magstripe
    ret, thresh1 = cv2.threshold(clahe_img, 35, 255, cv2.THRESH_BINARY)
    # Create edges around  image with auto min and max values
    img_canny = auto_canny(thresh1)
    # Sets a kernal 3*3 used for canny img_canny_dilation
    kernel = np.ones((3, 3), np.uint8)
    img_canny_dilation = cv2.dilate(img_canny, kernel, iterations=1)
    # saves frame with canny name to file path #Blocking off to increase render speed now returns image to be used
    #cv2.imwrite('filename.png', img_canny_dilation)
    return img_canny_dilation


def make_canny_face(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    img_canny = auto_canny(blurred_img)
    return img_canny


# Runs an automated version of canny with optimal lower and upper thresholds || sigma .33 Default
def auto_canny(img, sigma=0.33):
    # Gets median value of channel intensity | unique for each image
    v = np.median(img)
    # Sets lower int valeue for canny function
    lower = int(max(0, (1.0 - sigma) * v))
    # Sets higher int value for canny function
    upper = int(min(255, (1.0 + sigma) * v))
    # Img_canny now has the canny edge image
    img_canny = cv2.Canny(img, lower, upper)
    return img_canny
