import os
import sys
import numpy as np


# protocol://host:port/script_name?script_params|auth


from src.front_quantify import front_mm_metrics, front_mm_metrics_postmtrp
from src.side_quantify import side_mm_metrics, side_mm_metrics_postmtrp
from src.key_frame_extraction import split_frames
from src.video_to_mm import video_to_pixel_mm


def main(f_path, s_path):
    ear_to_ear_mm, head_width_mm  = front_mm_metrics(f_path)
    front2nape_mm, length_mm = side_mm_metrics(s_path)

    circumference_mm = (((head_width_mm * 2) + (length_mm * 2)) * 0.834626841674)

    return circumference_mm, front2nape_mm, ear_to_ear_mm, length_mm, head_width_mm


def run(path, ios):
    img_array = split_frames(path, ios)

    pixel_mm = video_to_pixel_mm(img_array)

    ear_to_ear_mm, head_width_mm  = front_mm_metrics_postmtrp(img_array, pixel_mm)

    front2nape_mm, length_mm = side_mm_metrics_postmtrp(img_array, pixel_mm)

    circumference_mm = int(((head_width_mm * 2) + (length_mm * 2)) * 0.834626841674)

    return circumference_mm, front2nape_mm, ear_to_ear_mm, length_mm, head_width_mm


#circumference_mm, front2nape_mm, ear_to_ear_mm, length_mm, head_width_mm = run('Videoq_Tests/Test_Video_from_Desma/jada-1.MOV')
#circumference_mm, front2nape_mm, ear_to_ear_mm, length_mm, head_width_mm = run('Video_Tests/Test_Video_from_Desma/avery-1.MOV')
#circumference_mm, front2nape_mm, ear_to_ear_mm, length_mm, head_width_mm = run('Video_Tests/Test_Video_from_Desma/desma-1.MOV')
#circumference_mm, front2nape_mm, ear_to_ear_mm, length_mm, head_width_mm = run('Video_Tests/Test_Video_from_Desma/james-1.MOV')
#circumference_mm, front2nape_mm, ear_to_ear_mm, length_mm, head_width_mm = run('Video_Tests/Test_Video_from_Desma/angie-1.MOV')
#circumference_mm, front2nape_mm, ear_to_ear_mm, length_mm, head_width_mm = run('Video_Tests/Test_Video_from_Desma/jay-1.MOV', ios=True)
circumference_mm, front2nape_mm, ear_to_ear_mm, length_mm, head_width_mm = run('Video_Tests/Self.mp4', ios = False)


print(circumference_mm, front2nape_mm, ear_to_ear_mm, length_mm, head_width_mm)
