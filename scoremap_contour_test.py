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
    image_path = 'D:\\dart\\cnt_test_images\\new_cam\\0.jpg'
    image = cv2.imread(image_path)
    cam = 2
    output_cam_dir = os.path.join(output_dir , str(cam) )
  
    config = params[cam]
    i=cam
    im = image.copy()
    min_area = 400
    threshold = 50
    
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
    black_background_im = result_image.copy()
    result_image[inverse_mask != 0] = 255


    brightness_level = get_brightness_level(image)
    if brightness_level == BRIGHTNESS_LEVEL.HIGH:
        threshold = 85
        min_area = 300
    elif brightness_level == BRIGHTNESS_LEVEL.MEDIUM:
        threshold = 65
    image = result_image
    color_corrected_image = make_black(image, threshold)

    bullseye_image, inner_be, outer_be = get_inner_outer_bullseye(image)
    print("inner be shape: ", inner_be.shape)
    cX, cY = get_center_coordinates(bullseye_image)

    bullseye_image, inner_be, outer_be = get_inner_outer_bullseye_by_coordinates(color_corrected_image, cX, cY)
    cX, cY = get_center_coordinates(bullseye_image)

    output_img_red, output_img_green = get_red_green_masks(color_corrected_image)
    #output_img_red = remove_small_contours(output_img_red, min_area)
    red_green_image = cv2.add(output_img_red, output_img_green)
    plt.imshow(red_green_image)
    plt.show()
    rings_image = crop_region(red_green_image, cY)
    plt.imshow(rings_image)
    plt.show()
    #notext_image = get_outer_border(rings_image, color_corrected_image, (cX, cY), threshold)
    inner_ring, outer_ring, inner_ring_cnt, outer_ring_cnt = get_inner_outer_rings(rings_image)
    plt.imshow(inner_ring)
    plt.show()
    plt.imshow(outer_ring)
    plt.show()
    #detected_regions = cv2.add(cv2.add(bullseye_image, outer_ring), inner_ring)
    #playing_ground = ROI_mask(outer_ring, outer_ring_cnt, -1, [cX, cY])
    #oi_mask = ROI_mask(outer_ring, outer_ring_cnt, -2, [cX, cY])
    #io_mask = ROI_mask(inner_ring, inner_ring_cnt, -1, [cX, cY])

    #roi = get_ROI_image(image, oi_mask, io_mask)
    #sectors = get_sector_gradients(roi, [cX, cY], inner_ring_cnt, outer_ring_cnt, config["THRESH_SAME_RING"], image, i)
    median_center = [cX, cY] #calculate_center_as_median_intersection_point(sectors, [cX, cY])
    cX, cY = median_center

    #sectors_sorted = sorted(sectors, key=lambda x: x["grad_avg"])

    #score_map = map_pts(sectors_sorted, config)

    plt.imsave('resultimage.jpg',result_image)

    output_img_red, output_img_green = get_red_green_masks(result_image)
    output_img_red = remove_small_contours(output_img_red, min_area)
    red_green_image = cv2.add(output_img_red, output_img_green)
    

    inner_ring, outer_ring, inner_ring_cnt, outer_ring_cnt = get_inner_outer_rings(red_green_image)

    white_pixels = np.all(inner_ring == [255, 255, 255], axis=-1)

   
    result_image = np.where(white_pixels[..., np.newaxis], [255, 255, 255], result_image)
    white_pixels = np.all(outer_ring == [255, 255, 255], axis=-1)

   
    result_image = np.where(white_pixels[..., np.newaxis], [255, 255, 255], result_image)
    white_pixels = np.all(inner_be == [255, 255, 255], axis=-1)

 
    result_image = np.where(white_pixels[..., np.newaxis], [255, 255, 255], result_image)
    white_pixels = np.all(outer_be == [255, 255, 255], axis=-1)

 
    result_image = np.where(white_pixels[..., np.newaxis], [255, 255, 255], result_image)


    result_image = result_image.astype(np.uint8)
    #plt.imshow(result_image)
    #plt.show()
    gray_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)

    h,w = gray_image.shape
                
    # Create a mask where 1 indicates the regions to be filtered and 0 indicates the regions to be unchanged
    mask = np.ones((h, w), dtype=np.uint8) * 255

    # Define the rectangular region at the center of the image
    # TODO use bulls eye here
    
    rect_start_row = (h - 200) // 2
    rect_end_row = rect_start_row + 200
    rect_start_col = (w - 300) // 2
    rect_end_col = rect_start_col + 300

    # Exclude the rectangular region from the mask
    cv2.rectangle(mask, (rect_start_col, rect_start_row), (rect_end_col, rect_end_row), 0, -1)
    # Apply median filter to the entire image, excluding the rectangular region
    filtered_image = cv2.medianBlur(gray_image, 7)

    # Restore the rectangular region in the filtered image
    gray_image = np.where(mask == 0, gray_image, filtered_image)

    
    
    #gray_image = cv2.medianBlur(gray_image, 9)
    threshold =70
    areas = []
    all_thresholds = []
    while threshold <190 : 
        
        _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY )
        contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=False)
        rem_contours = []
        area = 0 
        for i, contour in enumerate(sorted_contours[:-1]):
            if cv2.contourArea(contour) >1000:
                out_im = np.zeros_like(image)
                rem_contours.append(contour)
                area+= cv2.contourArea(contour)
                
        
        
        if len(rem_contours)== 20:
            areas.append(area)
            all_thresholds.append(threshold)
            #print(f'{threshold} {len(rem_contours)} {area}')
        threshold+=1
        out_im = cv2.drawContours(np.zeros((h,w), np.uint8), rem_contours, -1, 255,
                                thickness=cv2.FILLED)
        # plt.imshow( out_im)
        # plt.show()

    if len(areas)>0: 
        areas=np.array(areas)
        idx= np.argmax(areas)
        threshold = all_thresholds[idx]
        _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY )
        contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=False)
        black_contours = []
      
        for i, contour in enumerate(sorted_contours[:-1]):
            if cv2.contourArea(contour) >1000:
                black_contours.append(contour)



        #for white hereee      making pixels black for white transform
        out_im = black_background_im.copy()
        # out_im = cv2.drawContours(np.zeros((h,w), np.uint8), rem_contours, -1, 255,
        #                             thickness=cv2.FILLED)   

        black_background_im = cv2.drawContours(out_im, black_contours, -1, (0,0,0),
                                    thickness=cv2.FILLED)     
        #output_image_path =os.path.join(output_cam_dir , exp+'_'+str(len(rem_contours))+'_' +str(threshold)+ '.jpg' )
        
        #plt.imsave(output_image_path,out_im, cmap ='gray')
       # plt.imshow(out_im )
       # plt.show()
      
        white_pixels = np.all(inner_ring == [255, 255, 255], axis=-1)

    
        black_background_im = np.where(white_pixels[..., np.newaxis], [0, 0, 0], black_background_im)
        white_pixels = np.all(outer_ring == [255, 255, 255], axis=-1)

    
        black_background_im = np.where(white_pixels[..., np.newaxis], [0, 0, 0], black_background_im)
        white_pixels = np.all(inner_be == [255, 255, 255], axis=-1)

    
        black_background_im = np.where(white_pixels[..., np.newaxis], [0, 0, 0], black_background_im)
        white_pixels = np.all(outer_be == [255, 255, 255], axis=-1)

    
        black_background_im = np.where(white_pixels[..., np.newaxis], [0, 0, 0], black_background_im)
        # white image after removing inner outer ring be 
       # plt.imshow(black_background_im)
        #plt.show()



        gray_image = cv2.cvtColor(black_background_im.astype('uint8'), cv2.COLOR_BGR2GRAY)

        h,w = gray_image.shape
                    
        # Create a mask where 1 indicates the regions to be filtered and 0 indicates the regions to be unchanged
        mask = np.ones((h, w), dtype=np.uint8) * 255

        # Define the rectangular region at the center of the image
        # TODO use bulls eye here
        
        rect_start_row = (h - 200) // 2
        rect_end_row = rect_start_row + 200
        rect_start_col = (w - 300) // 2
        rect_end_col = rect_start_col + 300
 
        
        # Exclude the rectangular region from the mask
        cv2.rectangle(mask, (rect_start_col, rect_start_row), (rect_end_col, rect_end_row), 0, -1)
        # Apply median filter to the entire image, excluding the rectangular region
        filtered_image = cv2.medianBlur(gray_image, 9)

        # Restore the rectangular region in the filtered image
        gray_image = np.where(mask == 0, gray_image, filtered_image)

      
        #plt.imshow(gray_image , cmap= 'gray')
        plt.show()
        plt.imsave('white_gray.jpg',gray_image , cmap= 'gray')

        blurred = cv2.medianBlur(gray_image, 9)
        #plt.imshow(blurred)
        

        areas = []
        all_thresholds= [] 
        for threshold in range(70, 245):
            print(threshold)
            _, binary_image = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY )
            contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            sorted_contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=False)
            rem_contours = []
            area = 0 
            for i, contour in enumerate(sorted_contours[:]):
                if cv2.contourArea(contour) >800:
                    out_im = np.zeros_like(gray_image)
                    rem_contours.append(contour)
                    area+= cv2.contourArea(contour)
                    
            
            
            if len(rem_contours)== 20:
                areas.append(area)
                all_thresholds.append(threshold)
                print(f'{threshold} {len(rem_contours)} {area}')
            
                # out_im = cv2.drawContours(np.zeros((gray_image.shape), np.uint8), rem_contours, -1, 255,
                #                         thickness=cv2.FILLED)
                # plt.imshow( out_im)
                # plt.show()
        # plot white contours here
        if len(areas)>0: 
            areas=np.array(areas)
            idx= np.argmax(areas)
            threshold = all_thresholds[idx]
            _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY )
            contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            sorted_contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=False)
            white_contours = []
        
            for i, contour in enumerate(sorted_contours[:]):
                if cv2.contourArea(contour) >1000:
                    white_contours.append(contour)
                
            out_im = result_image.copy()
            # out_im = cv2.drawContours(np.zeros((h,w), np.uint8), rem_contours, -1, 255,
            #                             thickness=cv2.FILLED)   

            white_final = cv2.drawContours(out_im, white_contours, -1, (0,255,255),
                                        thickness=cv2.FILLED)     
        #output_image_path =os.path.join(output_cam_dir , exp+'_'+str(len(rem_contours))+'_' +str(threshold)+ '.jpg' )
        
        #plt.imsave(output_image_path,out_im, cmap ='gray')
            # plt.imshow(white_final )
            # plt.show()
            out_im = result_image.copy()
            # out_im = cv2.drawContours(np.zeros((h,w), np.uint8), rem_contours, -1, 255,
            #                             thickness=cv2.FILLED)   

            all_final = cv2.drawContours(out_im, white_contours, -1, (0,255,120),
                                        thickness=cv2.FILLED)     
            all_final = cv2.drawContours(all_final, black_contours, -1, (0,255,255),
                                        thickness=cv2.FILLED)
            # plt.imshow(all_final )
            # plt.show()

            black_centers = get_contours_centers(black_contours)
            black_bearings = get_bearings(black_centers , cX , cY)
            white_centers = get_contours_centers(white_contours)
            white_bearings = get_bearings(white_centers , cX , cY)
            black_lists = [(black_bearings[i], black_contours[i]) for i in range(len(black_bearings))]
            white_lists = [(white_bearings[i], white_contours[i]) for i in range(len(white_bearings))]
            sorted_combined = sorted(black_lists, key=lambda x: x[0])

            # Extract the sorted elements into separate lists
            black_bearings_sorted = [x[0] for x in sorted_combined]
            black_contours_sorted = [x[1] for x in sorted_combined]
            sorted_combined = sorted(white_lists, key=lambda x: x[0])

            # Extract the sorted elements into separate lists
            white_bearings_sorted = [x[0] for x in sorted_combined]
            white_contours_sorted = [x[1] for x in sorted_combined]
            scoremap_with_contour = [] 
            for contour, score, angle in zip(black_contours_sorted, config['BLACK_SECTORS'] , black_bearings_sorted ):
                obj  = (score, angle, contour )
                scoremap_with_contour.append(obj)
            for contour, score, angle in zip(white_contours_sorted, config['WHITE_SECTORS'] , white_bearings_sorted ):
                obj  = (score, angle, contour )
                scoremap_with_contour.append(obj)
            
            
            out_im = np.copy(im)
         
            print(len(scoremap_with_contour))
            for obj in scoremap_with_contour:
                score, angle, contour = obj
                if score in  config['WHITE_SECTORS']: 
                    cv2.fillPoly(out_im, [contour], color=(0,255,255))
                else:
                    cv2.fillPoly(out_im, [contour], color=(0,255,140))

                
                M = cv2.moments(contour)
                
                # Calculate contour center
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 0, 0  # Avoid division by zero   
                    
                cv2.putText(out_im,str(score), (cX , cY ),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
            plt.imshow(out_im)
            plt.show()
           
            
           
        # print(black_centers)
        # print( cX , cY)
        # #print(len(black_centers))
        


        # bearings = get_bearings(black_centers , cX , cY)
        # print(bearings)
        # print('contours len : ', len(rem_contours))
        # print('bearing len : ',len(bearings))
        # combined_lists = [(bearings[i], rem_contours[i]) for i in range(len(bearings))]

        # # Sort the list of tuples based on list1
        # sorted_combined = sorted(combined_lists, key=lambda x: x[0])

        # # Extract the sorted elements into separate lists
        # bearings_sorted = [x[0] for x in sorted_combined]
        # contours_sorted = [x[1] for x in sorted_combined]


        # #sorted_lists = sorted(zip(bearings, rem_contours))
        # #bearings_sorted, contours_sorted = zip(*sorted_lists)
        # print(bearings_sorted)
        # scoremap_with_contour = [] 
        # for contour, score, angle in zip(contours_sorted, config['BLACK_SECTORS'] , bearings_sorted ):
        #     obj  = (score, angle, contour )
        #     scoremap_with_contour.append(obj)

        # #print(scoremap_with_contour)
        # black_gradients = calculate_gradients(black_centers, (cX, cY))
        
        # black_obj , white_obj = [] , []
        # for idx , obj in enumerate(score_map):
        #     if obj[2] in params['BLACK_SECTORS']:
        #         black_obj.append(obj)
        # #print("black objects")
        # #print(black_obj)
        

        # # for point in black_centers:
        # #     bearing = calculate_bearing_from_north(point , cX , cY)
        # #     print(f"Bearing of point {point}: {bearing} degrees from north")



        # black_scoremap_contour = get_scoremap_with_contour(black_gradients , rem_contours, black_obj , (cX , cY),black_centers )         
        # #print("Scoremap")
        # #print(score_map)
        # #print(black_scoremap_contour[4])
        # #print(len(black_scoremap_contour))
        # out_im = np.copy(im)
        # black_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        # print(len(scoremap_with_contour))
        # for obj in scoremap_with_contour:
        #     score, angle, contour = obj
        #     #score =obj[2]
        #     #contours_dict =obj[3:]
            
        #     # for c in contours_dict:
        #     #     contour = c['contour']
        #         # out_im= cv2.drawContours(out_im, contour, -1, (0,0,0),
        #         #                         thickness=-1) 
                
                
        #     cv2.fillPoly(out_im, [contour], color=(0,255,255))

        #         # Apply the mask to the white image
        #         #out_im = cv2.bitwise_and(out_im, out_im, mask=black_mask)

        #     M = cv2.moments(contour)
    
        #     # Calculate contour center
        #     if M["m00"] != 0:
        #         cX = int(M["m10"] / M["m00"])
        #         cY = int(M["m01"] / M["m00"])
        #     else:
        #         cX, cY = 0, 0  # Avoid division by zero   
                
        #     cv2.putText(out_im,str(score), (cX , cY ),
        #         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        # # plt.imshow(out_im)
        # # plt.show()
        # #output_image_path =os.path.join(output_cam_dir , exp+'_'+str(len(rem_contours))+'_' +str(threshold)+ '.jpg' )
        
        # #plt.imsave(output_image_path,out_im)
        # plt.imshow(out_im)
        # plt.show()
