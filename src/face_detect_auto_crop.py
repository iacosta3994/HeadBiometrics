
def crop_above_eyes(img, eye_xy):
    if img is None:
        print("Image not loaded to crop")
        return None

    eye_x, eye_y = eye_xy

    height, width = img.shape[:2]

    # top left of image is 0,0 and bottom right is height, width
    beginX = 0
    endX = int(width)
    beginY = 0
    endY = int(eye_y)
    #crops image to keep everything above the Y location of the eye
    top_img = img[beginY:endY, beginX:endX]

    return top_img
