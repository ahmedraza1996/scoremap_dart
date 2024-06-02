import cv2
import numpy as np
import matplotlib.pyplot as plt
import os 
from autoscorer.dartboard_function import *
from autoscorer.dartboard_function import params as config 



if __name__ == '__main__':
    image_dir = 'D:/dart/pipeline_9May/board_images'
    output_dir = 'D:/dart/contour_outputs/iterative method'
    experiments= os.listdir(image_dir)
    # for exp in experiments: 
    #     for cam in [0,1,2,3]:
        
            # image_path =  os.path.join(image_dir, exp, 'board',f'{cam}.jpg')
    image_path = "D:\\dart\\pipeline_9May\\board_images\\b_120_20240328_135503_615t_120_20240328_140035_965\\board\\2.jpg"
    #image_path ="D:\\dart\singularity_new_cam\\b_SING_BOARD_20240513_132537_685t_SING_BOARD_20240513_132545_651\\board\\1.jpg"
    image = cv2.imread(image_path)
    cam = 2
    output_cam_dir = os.path.join(output_dir , str(cam) )
    
    config = params[cam]
    i=cam
    im = image.copy()
    model = YOLO('D://dart/best.pt')
    results = model.predict(source=im, show=False,
                        hide_labels=False, 
                        hide_conf= False,
                        save_txt=False,
                        save_crop=False,  
                        conf=0.25 , 
                        save= False )

    segmented_image = results[0].orig_img
    h, w, c =  image.shape
    blank_Im = np.zeros((h,w,1))
    polygons= results[0].masks.xy[0]
    ellipse = cv2.fitEllipse(polygons)
    ellipse_contour  = poly = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])), (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)), int(ellipse[2]), 0, 360, 5)
    ell_img= cv2.fillPoly(np.zeros_like(blank_Im), [ellipse_contour], 255)
    ell_img = np.uint8(ell_img)
    b, g, r = cv2.split(im)
    inverse_mask = cv2.bitwise_not(ell_img)
    # Subtract the mask from each channel
    b_result = cv2.subtract(b, inverse_mask)
    g_result = cv2.subtract(g, inverse_mask)
    r_result = cv2.subtract(r, inverse_mask)

    # Clip the resulting values to ensure they remain within the valid range
    b_result = np.clip(b_result, 0, 255)
    g_result = np.clip(g_result, 0, 255)
    r_result = np.clip(r_result, 0, 255)

    # Merge the channels back together
    result_image = cv2.merge((b_result, g_result, r_result))
    black_background_im = result_image
    result_image[inverse_mask != 0] = 255

    min_area = 400
    threshold = 50
    brightness_level = get_brightness_level(image)
    if brightness_level == BRIGHTNESS_LEVEL.HIGH:
        threshold = 85
        min_area = 300
    elif brightness_level == BRIGHTNESS_LEVEL.MEDIUM:
        threshold = 65

    color_corrected_image = make_black(result_image, threshold)

    bullseye_image, inner_be, outer_be = get_inner_outer_bullseye(result_image)
    print("inner be shape: ", inner_be.shape)
    cX, cY = get_center_coordinates(bullseye_image)

    bullseye_image, inner_be, outer_be = get_inner_outer_bullseye_by_coordinates(color_corrected_image, cX, cY)
    cX, cY = get_center_coordinates(bullseye_image)

    output_img_red, output_img_green = get_red_green_masks(color_corrected_image)
    #output_img_red = remove_small_contours(output_img_red, min_area)
    red_green_image = cv2.add(output_img_red, output_img_green)
    # plt.imshow(red_green_image)
    # plt.show()
    #rings_image = crop_region(red_green_image, cY)
  
    
  
    # inner_ring, outer_ring, inner_ring_cnt, outer_ring_cnt = get_inner_outer_rings(red_green_image)
    # plt.imshow(inner_ring)
    # plt.show()

    grayscale_image = cv2.cvtColor(red_green_image, cv2.COLOR_BGR2GRAY)
    # TODO: Change threshold
    _, threshold = cv2.threshold(grayscale_image, 50, 255,
                                 cv2.THRESH_BINARY)
    kernel = np.ones((3, 3))
    closing = cv2.dilate(threshold, kernel, iterations=5)
    closing = cv2.erode(closing, kernel, iterations=5)
    contours, hr = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print(" total contours: ", len(contours))
    sorted_contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=False)
    
    for c in sorted_contours :
       
        c_img=cv2.drawContours(np.zeros_like(grayscale_image), [c], -1,255, thickness=cv2.FILLED)  # cv2.FILLED)
        plt.imshow(c_img)
        plt.show()
    # inner_inside =sorted_contours[1]
    # inner_outside =sorted_contours[2]
    # outer_inside =sorted_contours[3]
    # outer_outside =sorted_contours[4]
    # inner_img = np.zeros_like(grayscale_image)
    # inner_img=cv2.drawContours(inner_img, [inner_outside], -1,255, thickness=cv2.FILLED)  # cv2.FILLED)
    # inner_img=cv2.drawContours(inner_img, [inner_inside], -1,0, thickness=cv2.FILLED)  # cv2.FILLED)
    
    
    # plt.imshow(inner_img)
    # plt.show()
    # outer_img = np.zeros_like(grayscale_image)
    # outer_img=cv2.drawContours(outer_img, [outer_outside], -1,255, thickness=cv2.FILLED)  # cv2.FILLED)
    # outer_img=cv2.drawContours(outer_img, [outer_inside], -1,0, thickness=cv2.FILLED)  # cv2.FILLED)
    
    
    # plt.imshow(outer_img)
    # plt.show()
    # new_im =grayscale_image.copy()
    # new_im[outer_img ==255]=255
    # plt.imshow(new_im)
    # plt.show()
    # closing_coloured = cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)
    # c_img = cv2.drawContours(closing_coloured.copy(), contours, -1, (255, 0, 0), thickness=1)  # cv2.FILLED)
    # contour_areas = [cv2.contourArea(contour, False) for contour in contours]
    # contours_sorted = sorted(contour_areas)
    # outer_ring_area = contours_sorted[-1]
    # inner_ring_area = contours_sorted[-2]
    # outer_ring_cnt = [contour for contour in contours if cv2.contourArea(contour, False) == outer_ring_area]
    # inner_ring_cnt = [contour for contour in contours if cv2.contourArea(contour, False) == inner_ring_area]
    # outer_ring = cv2.drawContours(np.zeros(closing_coloured.shape, np.uint8), outer_ring_cnt, -1, (255, 255, 255),
    #                               thickness=cv2.FILLED)
    # inner_ring = cv2.drawContours(np.zeros(closing_coloured.shape, np.uint8), inner_ring_cnt, -1, (255, 255, 255),
    #                               thickness=cv2.FILLED)