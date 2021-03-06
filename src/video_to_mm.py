import numpy as np
import cv2
from src.canny_edge_detection_cv2 import make_canny_magstripe
from src.mag_stripe_search import get_magstripe_demensions
from src.standard_dev_filter import std_filter
from src.remove_nested import remove_nested_with_idx


# Funtion that inputs video mp4, name of scene with location, variable to determine how many frames to splice


def video_to_pixel_mm(split_frame_array):

    frame_proccessed = 0

    # Later this list will have all the H and W of the mag stripe
    list_mag_stripe_w_h_area_ratio_magname = []

    while (len(split_frame_array)) >= frame_proccessed:
        for frame in split_frame_array:

            # Canny function for magstripe benchmark
            canny_image = make_canny_magstripe(frame)

            # canny_image is then used with get_magstripe_demensions
            frame_mag_stripe_w_h_ar_mn_xy_list = get_magstripe_demensions(canny_image)

            for mag_stripe_w_h_area_ratio_magname in frame_mag_stripe_w_h_ar_mn_xy_list:
                # It checks if any width and height was in the var
                if mag_stripe_w_h_area_ratio_magname:
                    # Appends data into list mag stripe to be consolidated after it finishes with loop
                    list_mag_stripe_w_h_area_ratio_magname.append(mag_stripe_w_h_area_ratio_magname)
                    # Ticks the counter for the next frame
            frame_proccessed += 1

            # Once all the frames have been processed continue below

    area_std_filter_list = std_filter(list_mag_stripe_w_h_area_ratio_magname, 2, 3)
    list_mag_stripe_filtered = remove_nested_with_idx(
        list_mag_stripe_w_h_area_ratio_magname, area_std_filter_list, 2)

    aspr_std_filter_list = std_filter(list_mag_stripe_filtered, 3, 3)
    list_mag_stripe_filtered = remove_nested_with_idx(
        list_mag_stripe_filtered, aspr_std_filter_list, 3)

    # diffrent magstripes demensions that are manufactured
    mag_stripe_constant_w_h_1 = [84.40, 7.9375]
    mag_stripe_constant_w_h_2 = [84.40, 8.382]
    mag_stripe_constant_w_h_3 = [84.40, 12.7]
    mag_stripe_constant_w_h_4 = [84.40, 13.48]

    mag_stripe_w_h = []
    pixel_mm_mean_list = []
    #dp_magdp_list = []

    for mag_stripe_w_h_area_ratio_magname in list_mag_stripe_filtered:

        if 0.08981481481 <= mag_stripe_w_h_area_ratio_magname[3] <= 0.095:
            mag_stripe_w_h = mag_stripe_w_h_area_ratio_magname[:2]
            dp = np.sqrt(mag_stripe_w_h[0]**2 + mag_stripe_w_h[1]**2)
            magstripe_diag = np.sqrt(
                mag_stripe_constant_w_h_2[0]**2 + mag_stripe_constant_w_h_2[1]**2)
            ppmm = magstripe_diag/dp
            #dp_magdp_list.append([dp, magstripe_diag])
            pixel_mm_mean_list.append([ppmm])

        elif 0.095 <= mag_stripe_w_h_area_ratio_magname[3] <= 0.102666585:
            mag_stripe_w_h = mag_stripe_w_h_area_ratio_magname[:2]
            dp = np.sqrt(mag_stripe_w_h[0]**2 + mag_stripe_w_h[1]**2)
            magstripe_diag = np.sqrt(
                mag_stripe_constant_w_h_2[0]**2 + mag_stripe_constant_w_h_2[1]**2)
            ppmm = magstripe_diag/dp
            #dp_magdp_list.append([dp, magstripe_diag])
            pixel_mm_mean_list.append([ppmm])

        elif 0.12314814814 <= mag_stripe_w_h_area_ratio_magname[3] <= 0.1385:
            mag_stripe_w_h = mag_stripe_w_h_area_ratio_magname[:2]
            dp = np.sqrt(mag_stripe_w_h[0]**2 + mag_stripe_w_h[1]**2)
            magstripe_diag = np.sqrt(
                mag_stripe_constant_w_h_3[0]**2 + mag_stripe_constant_w_h_3[1]**2)
            ppmm = magstripe_diag/dp
            #dp_magdp_list.append([dp, magstripe_diag])
            pixel_mm_mean_list.append([ppmm])

        elif 0.1386 <= mag_stripe_w_h_area_ratio_magname[3] <= 0.1573:
            mag_stripe_w_h = mag_stripe_w_h_area_ratio_magname[:2]
            dp = np.sqrt(mag_stripe_w_h[0]**2 + mag_stripe_w_h[1]**2)
            magstripe_diag = np.sqrt(
                mag_stripe_constant_w_h_4[0]**2 + mag_stripe_constant_w_h_4[1]**2)
            ppmm = magstripe_diag/dp
            #dp_magdp_list.append([dp, magstripe_diag])
            pixel_mm_mean_list.append([ppmm])

        else:
            continue

    '''with open('magstripe_pixel_w_h_a_n.txt', 'w') as file_txt:
            for w_h_ar_n in list_mag_stripe_filtered:
                file_txt.write(str(w_h_ar_n) + ' ' + '\n')

        with open('dp_magdp.txt', 'w') as file_txt:
            for dp_magdp in dp_magdp_list:
                file_txt.write(str(dp_magdp) + ' ' + '\n')

        with open('magstripe_ppmm_results.txt', 'w') as file_txt:
            for ppmm in pixel_mm_mean_list:
                file_txt.write(str(ppmm) + ' ' + '\n') '''



    pixel_mm_filter = std_filter(pixel_mm_mean_list, 0, deviations_count=3)
    pixel_mm_filtered = remove_nested_with_idx(pixel_mm_mean_list, pixel_mm_filter, 0)

    return min(pixel_mm_filtered)
