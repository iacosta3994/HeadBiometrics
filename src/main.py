import os
import sys
import numpy as np

from src.front_quantify import front_mm_metrics_postmtrp
from src.side_quantify import side_mm_metrics_postmtrp
from src.key_frame_extraction import split_frames
from src.video_to_mm import video_to_pixel_mm
from src.narrow_wide_img_select import narrow_wide_img

def run(path, clockwise):
    img_array = split_frames(path, clockwise)

    pixel_mm = video_to_pixel_mm(img_array)

    narrow_img, wide_img = narrow_wide_img(img_array)

    ear_to_ear_mm, head_width_mm  = front_mm_metrics_postmtrp(narrow_img, pixel_mm)

    front2nape_mm, length_mm = side_mm_metrics_postmtrp(wide_img, pixel_mm)

    circumference_mm = int(((head_width_mm * 2) + (length_mm * 2)) * 0.834626841674)

    return circumference_mm, front2nape_mm, ear_to_ear_mm


#Test program demostrating the program
circumference_mm, front2nape_mm, ear_to_ear_mm = run('Video_Tests/Self.mp4', clockwise = False)
print(circumference_mm, front2nape_mm, ear_to_ear_mm)
