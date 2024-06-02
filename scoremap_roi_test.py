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
    image_path = 'D:\\dart\\pipeline_9May\\board_images\\b_120_20240328_135503_615t_120_20240328_140035_965\\board\\2.jpg'
    image = cv2.imread(image_path)
    cam = 2
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

    bullseye_image, inner_be, outer_be = get_inner_outer_bullseye(image)
    print("inner be shape: ", inner_be.shape)
    cX, cY = get_center_coordinates(bullseye_image)

    bullseye_image, inner_be, outer_be = get_inner_outer_bullseye_by_coordinates(color_corrected_image, cX, cY)
    cX, cY = get_center_coordinates(bullseye_image)

    output_img_red, output_img_green = get_red_green_masks(color_corrected_image)
    output_img_red = remove_small_contours(output_img_red, min_area)
    red_green_image = cv2.add(output_img_red, output_img_green)
    rings_image = crop_region(red_green_image, cY)
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

    sectors_sorted = sorted(sectors, key=lambda x: x["grad_avg"])

    score_map = map_pts(sectors_sorted, config)

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

    plt.imsave('resultimage.jpg',result_image)

    

    black_centers = get_contours_centers(rem_contours)
    print(black_centers)
    print( cX , cY)
    #print(len(black_centers))
    


    bearings = get_bearings(black_centers , cX , cY)
    print(bearings)
    print('contours len : ', len(rem_contours))
    print('bearing len : ',len(bearings))
    combined_lists = [(bearings[i], rem_contours[i]) for i in range(len(bearings))]

    # Sort the list of tuples based on list1
    sorted_combined = sorted(combined_lists, key=lambda x: x[0])

    # Extract the sorted elements into separate lists
    bearings_sorted = [x[0] for x in sorted_combined]
    contours_sorted = [x[1] for x in sorted_combined]


    #sorted_lists = sorted(zip(bearings, rem_contours))
    #bearings_sorted, contours_sorted = zip(*sorted_lists)
    print(bearings_sorted)
    scoremap_with_contour = [] 
    for contour, score, angle in zip(contours_sorted, config['BLACK_SECTORS'] , bearings_sorted ):
        obj  = (score, angle, contour )
        scoremap_with_contour.append(obj)

    #print(scoremap_with_contour)
    black_gradients = calculate_gradients(black_centers, (cX, cY))
    
    black_obj , white_obj = [] , []
    for idx , obj in enumerate(score_map):
        if obj[2] in params['BLACK_SECTORS']:
            black_obj.append(obj)
    #print("black objects")
    #print(black_obj)
    

    # for point in black_centers:
    #     bearing = calculate_bearing_from_north(point , cX , cY)
    #     print(f"Bearing of point {point}: {bearing} degrees from north")



    black_scoremap_contour = get_scoremap_with_contour(black_gradients , rem_contours, black_obj , (cX , cY),black_centers )         
    #print("Scoremap")
    #print(score_map)
    #print(black_scoremap_contour[4])
    #print(len(black_scoremap_contour))
    out_im = np.copy(im)
    black_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    print(len(scoremap_with_contour))
    for obj in scoremap_with_contour:
        score, angle, contour = obj
        #score =obj[2]
        #contours_dict =obj[3:]
        
        # for c in contours_dict:
        #     contour = c['contour']
            # out_im= cv2.drawContours(out_im, contour, -1, (0,0,0),
            #                         thickness=-1) 
            
            
        cv2.fillPoly(out_im, [contour], color=(0,255,255))

            # Apply the mask to the white image
            #out_im = cv2.bitwise_and(out_im, out_im, mask=black_mask)

        M = cv2.moments(contour)

        # Calculate contour center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0  # Avoid division by zero   
            
        cv2.putText(out_im,str(score), (cX , cY ),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    # plt.imshow(out_im)
    # plt.show()
    #output_image_path =os.path.join(output_cam_dir , exp+'_'+str(len(rem_contours))+'_' +str(threshold)+ '.jpg' )
    
    #plt.imsave(output_image_path,out_im)
    plt.imshow(out_im)
    plt.show()
