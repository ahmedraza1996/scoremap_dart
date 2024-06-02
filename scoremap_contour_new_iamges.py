import cv2
import numpy as np
import matplotlib.pyplot as plt
import os 
from autoscorer.dartboard_function import *
from autoscorer.dartboard_function import params as config 



if __name__ == '__main__':
    image_dir = 'D:\\dart\\cnt_test_images\\board_tests'
    output_dir = 'D:\\dart\\contour_outputs\\cam_test2'
    experiments= os.listdir(image_dir)
    failed_exp =[]
    rows, cols = 2, 3
    img_height, img_width = 1080,1920
# Create the figure and axes objects
        
    for exp in experiments: 
       
       
        #axes = axes.reshape(rows, cols)
        try:
            image_path =  os.path.join(image_dir,exp)
            #image_path = "D:\\dart\\pipeline_9May\\board_images\\b_120_20240328_135503_615t_120_20240328_140035_965\\board\\2.jpg"
            image = cv2.imread(image_path)
            img_height, img_width, c = image.shape
            fig_width = cols * (img_width / 100)  # Convert pixels to inches (assuming 100 dpi)
            fig_height = rows * (img_height / 100)  # Convert pixels to inches (assuming 100 dpi)

            # Create the figure and axes objects
            fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

            cam = 0
            output_cam_dir = os.path.join(output_dir , str(cam) )

            config = params[cam]
            i=cam
            im = image.copy()
          
            min_area = 400
            threshold = 50
            brightness_level = get_brightness_level(image)
            if brightness_level == BRIGHTNESS_LEVEL.HIGH:
                threshold = 85
                min_area = 300
            elif brightness_level == BRIGHTNESS_LEVEL.MEDIUM:
                threshold = 65

            color_corrected_image = make_black(image, threshold)
            # plt.imshow(color_corrected_image)
            # plt.show()
            bullseye_image, inner_be, outer_be = get_inner_outer_bullseye(image)
            # plt.imshow(bullseye_image)
            # plt.show()
            #print("inner be shape: ", inner_be.shape)
            cX, cY = get_center_coordinates(bullseye_image)

            bullseye_image, inner_be, outer_be = get_inner_outer_bullseye_by_coordinates(color_corrected_image, cX, cY)
            # plt.imshow(bullseye_image)
            # plt.show()
            cX, cY = get_center_coordinates(bullseye_image)

            output_img_red, output_img_green = get_red_green_masks(color_corrected_image)
            # plt.imshow(output_img_red)
            # plt.show()
            # plt.imshow(output_img_green)
            # plt.show()
            output_img_red = remove_small_contours(output_img_red, min_area)
            red_green_image = cv2.add(output_img_red, output_img_green)
            # plt.imshow(red_green_image)
            # plt.show()
            rings_image = crop_region(red_green_image, cY)
            # plt.imshow(rings_image)
            # plt.show()
            notext_image = get_outer_border(rings_image, color_corrected_image, (cX, cY), threshold)
            inner_ring, outer_ring, inner_ring_cnt, outer_ring_cnt = get_inner_outer_rings(notext_image)
            detected_regions = cv2.add(cv2.add(bullseye_image, outer_ring), inner_ring)
            playing_ground = ROI_mask(outer_ring, outer_ring_cnt, -1, [cX, cY])
            oi_mask = ROI_mask(outer_ring, outer_ring_cnt, -2, [cX, cY])
            io_mask = ROI_mask(inner_ring, inner_ring_cnt, -1, [cX, cY])

            roi = get_ROI_image(image, oi_mask, io_mask)
            sectors = get_sector_gradients(roi, [cX, cY], inner_ring_cnt, outer_ring_cnt, config["THRESH_SAME_RING"], image, i)
            median_center = [cX, cY] #calculate_center_as_median_intersection_point(sectors, [cX, cY])
            cX, cY = median_center
            # plt.imshow(roi)
            # plt.show()
            #sectors_sorted = sorted(sectors, key=lambda x: x["grad_avg"])

            #score_map = map_pts(sectors_sorted, config)





                
            black_bg_image , white_bg_image = get_dartboard_roi(im)
            # plt.imshow(white_bg_image)
            # plt.show()
            output_img_red, output_img_green = get_red_green_masks(white_bg_image)
            output_img_red = remove_small_contours(output_img_red, min_area)
            red_green_image = cv2.add(output_img_red, output_img_green)
            # plt.imshow(red_green_image)
            # plt.show()

            inner_ring, outer_ring, inner_ring_cnt, outer_ring_cnt = get_inner_outer_rings(red_green_image)
            
            ir_image = cv2.add(inner_ring, outer_ring)
            # plt.imshow(ir_image)
            # plt.show()
            im=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            axes[0,0].imshow(im)
            axes[0,0].set_title(f"Image-{exp}")
            black_bg_image=cv2.cvtColor(black_bg_image, cv2.COLOR_BGR2RGB)
            axes[0,1].imshow(black_bg_image)
            axes[0,1].set_title(f"ROI black")
            white_bg_image=cv2.cvtColor(white_bg_image, cv2.COLOR_BGR2RGB)
            axes[0,2].imshow(white_bg_image)
            axes[0,2].set_title(f"ROI white")
            red_green_image=cv2.cvtColor(red_green_image, cv2.COLOR_BGR2RGB)
            axes[1,0].imshow(red_green_image)
            axes[1,0].set_title(f"Red green masks")
            ir_image=cv2.cvtColor(ir_image, cv2.COLOR_BGR2RGB)
            axes[1,1].imshow(ir_image)
            axes[1,1].set_title(f"Rings contours")
            axes[1,2].imshow(bullseye_image)
            axes[1,2].set_title(f"Bulls eye")
            
            plt.subplots_adjust(wspace=0.1, hspace=0.4)
            plt.savefig(f'{output_cam_dir}/{exp}_canvas.png', bbox_inches='tight')
            # # plt.imshow(white_roi_im)
            # # plt.show()
            # gray_image = preproces_image(white_roi_im)
            # # plt.imshow(gray_image)
            # # plt.show()
            # black_bg_image, black_contours = get_black_contour(gray_image , black_bg_image)
            # out_im = black_bg_image.copy()
            

            # out_im = cv2.drawContours(np.zeros_like(black_bg_image), black_contours, -1, (255,255,2555),
            #                             thickness=cv2.FILLED)  
            # # plt.imshow(out_im)
            # # plt.show()
            # black_roi_im = subtract_roi(black_bg_image,inner_ring, outer_ring , inner_be ,outer_be ,is_white=False )
            # gray_image = preproces_image(black_roi_im)
            # blurred = cv2.medianBlur(gray_image, 9)
            # white_contours = get_white_contour(blurred)
            
            
            #     # h,w,c = black_bg_image.shape
            #     # inner_ring_mask , outer_ring_mask = get_ring_mask(black_bg_image.copy() , white_contours, black_contours, inner_be , outer_be,cX,cY)
            #     # print(inner_ring_mask.shape)
            #     # multiplier_map = get_multiplier_map(inner_ring_mask, outer_ring_mask , h, w)
            #     # plt.imshow(multiplier_map)
            #     # plt.show()
            # out_im = blurred.copy()
            

            # out_im = cv2.drawContours(np.zeros_like(blurred), white_contours, -1, (255,255,2555),
            #                             thickness=cv2.FILLED)  
            # # plt.imshow(out_im)
            # # plt.show()

            # black_centers = get_contours_centers(black_contours)
            # black_bearings ,black_coords = get_bearings(black_centers , cX , cY)
            # print("black bearings", black_bearings)
            # white_centers = get_contours_centers(white_contours)
            # white_bearings , white_coords = get_bearings(white_centers , cX ,cY)
            # print("white bearings", white_bearings)
            # black_contours , black_bearings, white_contours, white_bearings, black_coords, white_coords= sorts_contour_with_bearings(black_contours , black_bearings, white_contours, white_bearings , black_coords, white_coords)
            # scoremap_with_contour = [] 
            # for contour, score, angle, black_coord in zip(black_contours, config['BLACK_SECTORS'] , black_bearings , black_coords ):
            #     obj  = (score, angle, contour , black_coord )
            #     scoremap_with_contour.append(obj)
            # for contour, score, angle , white_coord in zip(white_contours, config['WHITE_SECTORS'] , white_bearings , white_coords ):
            #     obj  = (score, angle, contour , white_coord)
            #     scoremap_with_contour.append(obj)
            # if not os.path.exists(output_cam_dir):
            #     os.makedirs(output_cam_dir)
            # save_scoremap_with_contour(im , scoremap_with_contour, config, os.path.join(output_cam_dir, exp+'.jpg'))
        except:
            failed_exp.append(exp)

    print(failed_exp)