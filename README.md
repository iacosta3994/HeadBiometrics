This is the Head biometrics program, Utilizing numpy, OpenCV, and TensorFlow libraries to calculate the dimensions of the head using a cellphone camera.

First we start off with Split Frames
Split frames input the incoming video file, The video frames are rotated 90 degrees clockwise or counterclockwise depending on the tags to ensure that the nupy array of frames are in portrait mode. This array will be used in 2 different function s to extract the head’s dimensions presented in the video. 

Show turning IMG clockwise or counterclockwise
Video to Pixel mm
This module is composed of 4 built modules, not including cv2 and NumPy.

Make Canny Magstripe
With the incoming array of frames, each image goes through the make canny magstripe function that creates an edge image by converting it to grayscale, blurring, and increase its contrast. The image goes through a binary threshold process to keep the darkest pixels in the image to help detect the magstripe in the picture. 

Show magstripe self.zip

Get Magstripe Dimensions 
The contours of the image are then extracted and follow into the detect magstripe function. 
This function goes through each contour, accounts for minimal glare, and generates a bounding box around the detected contour. The ones that closely match the aspect ratio expected are appended to a list of detected magstripes with information such as width, height, area, aspect ratio, along with a unique name

Std filter

The standard deviation filter allows for the list of detected magstripe to be filtered in accordance with area, and aspect ratio with the option to adjust the level of deviations to let through. 

IDX Filter
Then the final list goes throught he index filter and removes any detected magstripes that do not meet criteria

In the end, the smallest variable is returned indicating the real-world representation that makes up a pixel in the image.

Narrow, wide IMG select

In this module, the face detector model is loaded from the proctoring AI that uses Tensorflow and the facial feature landmark model. 

Each of the six facial landmark points is preestablished in an array. Points tracked include nose , chin, the corners of the left and right eye, and the left and right sides of the mouth.


Show the head pose estimator program.

Variables are established to track the horizontal and vertical angle of the head for the head pose estimation algorithm. For every image in the array all the points are collected, two images are saved, one where the face is closest to a neutral pose. The second image is then selected for which the head pose is facing farthest away from the neutral starting point. The image where the person faces directly which is the narrowest pint of the head is put through an automated cropping function using the eyes’ points for reference. It is set up to maintain self adaptability in changes of image resolution. 

Show widest IMG and narrowest, then show narrowest IMG after crop above eyes.

Front quantify
Import numpy cv2

This function has two unique modules that will be used o automatically complete the ear to ear and head width metric. 

NED front

The narrowest IMG is then fed into the neural edge detector model that utilizes TensorFlow to distinguish between the foreground and background. 

Show ned image

The contour of the head is then returned for the next steps. 

Ret contour width

This module analyzes the contour and generates a bounding box to determine the head above’s height and width above the eyes. Due to the ned image dilation process of the lines, the head’s width is offset by 50pts on each side following the dilation used in NED  to return the correct width of the contour. 

Front mm metrics 
The pixel width of the head and the ear to ear measurement are then multiplied by the pixel mm metric to return the measurements. 
Side Quantify
Numpy cv2 
Side quantify starts off with the point return function that gives the pixel coordinates to calculate the length of the head, and the front of the wanted hearline to the nape. 

The length is calculated by measuring the distance between the front and back points. 

To calculate for the front to nape, a couple of image manipulation processes are followed, the side image undergoes edge detection using canny and Sobel functions.  

Show canny image

The Sobel function uses a uses both an x-axis and y-axis filter. The grad x Sobel’s kernel size is 5 times larger to reduce the chances for breaks in the contour line primarily on the top of the head. The two images are then merged, the lines are dilated and eroded for a cleaner line. 

Using the 2 points for the front to nape, the program then deletes the pixels underneath the points leaving behind the top of the head’s contour used for the measurement. 

The contour is composed of 4 different sides, the outer top, the inner bottom, and the left and right edge. The contour used for the front to nape is only focused on the contour’s outer side, negating the other three. 

That contour length is converted to millimeters. 

The circumference is then calculated using an elliptical constant utilizing the height and width collected previously, 
In the end, we can return the circumference, front to nape, ear to ear, length, and width of the head in millimeters 
