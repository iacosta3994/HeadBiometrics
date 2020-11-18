import os
import sys
import numpy as np


# protocol://host:port/script_name?script_params|auth


from src.front_quantify import front_mm_metrics
from src.side_quantify import side_mm_metrics


def main(f_path, s_path):
    ear_to_ear_mm, head_width_mm  = front_mm_metrics(f_path)
    front2nape_mm, length_mm = side_mm_metrics(s_path)

    circumference_mm = (((head_width_mm * 2) + (length_mm * 2)) * 0.834626841674)

    return ear_to_ear_mm, front2nape_mm, circumference_mm


ear_to_ear_mm, front2nape_mm, circumference_mm = main(
    'Video_Tests/Test_Video_from_Desma/angie-1.MOV', 'Video_Tests/Test_Video_from_Desma/angie-1.MOV')
print(ear_to_ear_mm, front2nape_mm, circumference_mm)
