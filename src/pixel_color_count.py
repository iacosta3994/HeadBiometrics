import cv2
import numpy as np

img_name = input("What's the name of the picture? ")


img = cv2.imread(img_name)

# boundaries for the color red
boundaries_red = [
    ([37, 37, 50], [80, 80, 255])
]
boundaries_green = [
    ([90, 100, 90], [120, 255, 120])
]
boundaries_blue = [
    ([80, 0, 0], [255, 50, 50])
]

for(lower, upper) in boundaries_red:
    # creates numpy array from boundaries
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    # finds colors in boundaries a applies a mask
    mask = cv2.inRange(img, lower, upper)
    output = cv2.bitwise_and(img, img, mask=mask)

    # saves the image
    cv2.imwrite('2'+img_name, output)

tot_pixel = output.size
red_pixel = np.count_nonzero(output)
percentage = round(red_pixel * 100 / tot_pixel, 2)

print("color pixels: " + str(red_pixel))
print("Total pixels: " + str(tot_pixel))
print("Percentage of red pixels: " + str(percentage) + "%")
