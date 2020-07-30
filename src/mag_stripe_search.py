'''
Inputs will be a database of images, that have already been modified with canny.
Function will look for a set of pixels with same ratio demensions as mag_stripe.

With the image, Height and Width (pixel_H, pixel_W : 0 , 0)will be saved along with specific frameIdx it belongs to
This will help determine wich frame has the Highest Height & Width, new py that esablishes collection of values to find Highest values will be output
'''

'''
Parameters:	preCornerDetect
src – Source single-channel 8-bit of floating-point image.
dst – Output image that has the type CV_32F and the same size as src .
ksize – Aperture size of the Sobel() .
borderType – Pixel extrapolation method. See borderInterpolate() .
'''

cv2.preCornerDetect(src, ksize[, dst[, borderType]])


'''
Parameters:	cornerEigenValsAndVecs
src – Input single-channel 8-bit or floating-point image.
dst – Image to store the results. It has the same size as src and the type CV_32FC(6) .
blockSize – Neighborhood size (see details below).
ksize – Aperture parameter for the Sobel() operator.
borderType – Pixel extrapolation method. See borderInterpolate() .
'''
cv2.cornerEigenValsAndVecs(src, blockSize, ksize[, dst[, borderType]])
