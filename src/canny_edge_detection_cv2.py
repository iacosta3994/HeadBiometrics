import cv2
import numpy as np
from src.HED_model.edge_detection import CropLayer, args

# Function will load frame from file_path and will save canny edge image in same directory with new name


def make_canny_magstripe(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blurring to reduce artifacts
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    clahe = cv2.createCLAHE()  # Contrast Limited Adaptive Histogram Equalization
    # Adjsuts contrast in image to allow maggstripe to be darker for images with high gamma
    clahe_img = clahe.apply(blurred_img)
    # 35 works well Sets a threshold in search for the pixels near the color of black i.e magstripe
    ret, thresh1 = cv2.threshold(clahe_img, 85, 255, cv2.THRESH_BINARY)
    # Create edges around  image with auto min and max values
    img_canny = cv2.Canny(thresh1, 85, 255)
    # Sets a kernal 3*3 used for canny img_canny_dilation
    kernel = np.ones((3, 3), np.uint8)
    img_canny_dilation = cv2.dilate(img_canny, kernel, iterations=1)
    # saves frame with canny name to file path #Blocking off to increase render speed now returns image to be used
    #cv2.imwrite('filename.png', img_canny_dilation)
    return img_canny_dilation

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

    # faces uses sobel for the edge detection along the x and y axis and combines the img


def make_sobel_face(img):
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    # makes sure that img is already binary, if its a color photo it will turn it binary
    if len(img.shape) != 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # makes the gradient using x and y axis
    grad_x = cv2.Sobel(img, ddepth, 1, 0, ksize=25, scale=scale,
                       delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(img, ddepth, 0, 1, ksize=5, scale=scale,
                       delta=delta, borderType=cv2.BORDER_DEFAULT)
    # this section prepares it to combine together
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    img = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    # the white pixels are expanded to company any contours that are seperated
    img = cv2.dilate(img, (15, 15))

    kernel = np.ones((25,25),np.uint8)
    # runs another level of dilation and then erodes it
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    kernel = np.ones((11,11),np.uint8)
    img = cv2.erode(img, kernel, 1)
    return img


def auto_canny_face(img, sigma = 0.33):

    img = cv2.GaussianBlur(img, (5,5), 0)
    img = auto_canny(img, sigma)

    return img

def get_contour(contours):
    if len(contours) == 2:
        contour = contours[0]
    elif len(contours) == 3:
        contour = contours[1]
    else:
        raise Exception(
            ("contour in face_contour_width is not working as expected,findContours changed output type check opencv documentation for updates"))
    return contour





def neural_edge_detection(frame, ret_contour = True):


    cv2.dnn_registerLayer('Crop', CropLayer)
    # Load the model.
    net = cv2.dnn.readNet(args.prototxt, args.caffemodel)

    frame = cv2.GaussianBlur(frame, (9,9), 0)

    inp = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(args.width, args.height),
                               mean=(104.00698793, 116.66876762, 122.67891434),
                               swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()
    out = out[0, 0]
    out = cv2.resize(out, (frame.shape[1], frame.shape[0]))
    out = 255 * out
    out = out.astype(np.uint8)

    kernel = np.ones((15,15), np.uint8)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
    iterations = 1
    out = cv2.erode(out, kernel, iterations)


    if ret_contour is True:

        contours = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # grabs the largest contour

        contour = get_contour(contours)

        if len(contour) == 0:
            return None , None

        contour = max(contour, key = cv2.contourArea)

        return contour, out

    if ret_contour is False:
        return out
