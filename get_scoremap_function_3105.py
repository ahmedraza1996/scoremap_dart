




def get_score_map(image, config, i, roi_model):
    cam_dict ={}
    start_time = time.time()
    start_time_all = time.time()
    min_area = 400
    threshold = 50
    im = image.copy()
    brightness_level = get_brightness_level(image)
    if brightness_level == BRIGHTNESS_LEVEL.HIGH:
        threshold = 85
        min_area = 300
    elif brightness_level == BRIGHTNESS_LEVEL.MEDIUM:
        threshold = 65

    color_corrected_image = make_black(image, threshold)

    bullseye_image, inner_be, outer_be = get_inner_outer_bullseye(image)
    
    cX, cY = get_center_coordinates(bullseye_image)

    bullseye_image, inner_be, outer_be = get_inner_outer_bullseye_by_coordinates(color_corrected_image, cX, cY)
    
    cX, cY = get_center_coordinates(bullseye_image)
    end_time = time.time()
  
    black_bg_image , white_bg_image, roi_mask = get_dartboard_roi(im, model= roi_model)
    plt.imsave('/app/tmp/black_roi.jpg', black_bg_image)
    plt.imsave('/app/tmp/white_roi.jpg', white_bg_image)
 
    output_img_red, output_img_green = get_red_green_masks(black_bg_image)
    output_img_red = remove_small_contours(output_img_red, min_area)
    red_green_image = cv2.add(output_img_red, output_img_green)
    plt.imsave("red_green.jpg",red_green_image )
    rings_image = crop_region(red_green_image, cY)
    plt.imsave("rings_image.jpg",rings_image )
    #rings_image_lower, rings_image_upper = crop_region_two_parts(red_green_image , cY)
   
    notext_image = get_outer_border(rings_image, color_corrected_image, (cX, cY), threshold)
    plt.imsave("rings_image_notext.jpg",notext_image )
    inner_ring, outer_ring, inner_ring_cnt, outer_ring_cnt = get_inner_outer_rings(notext_image)
    # inner_ring_l, outer_ring_l, inner_ring_cnt_l, outer_ring_cnt_l = get_inner_outer_rings(rings_image_lower)

    # outer_ring_u, inner_ring_u, outer_ring_cnt_u, inner_ring_cnt_u = get_inner_outer_rings(rings_image_upper)
    # h,w,c = image.shape
    # inner_ring_full  = np.zeros((h,w))
    # outer_ring_full  = np.zeros((h,w))
    # inner_ring_full = cv2.drawContours(inner_ring_full, inner_ring_cnt_l+inner_ring_cnt_u, -1,255, -1)
    # outer_ring_full = cv2.drawContours(outer_ring_full, outer_ring_cnt_l+outer_ring_cnt_u, -1,255, -1)
   
    plt.imsave("full_outer.jpg",outer_ring )
    plt.imsave("full_inner.jpg",inner_ring )
    #detected_regions = cv2.add(cv2.add(bullseye_image, outer_ring), inner_ring)
    #playing_ground = ROI_mask(outer_ring, outer_ring_cnt, -1, [cX, cY])
   # oi_mask = ROI_mask(outer_ring, outer_ring_cnt, -2, [cX, cY])
    #io_mask = ROI_mask(inner_ring, inner_ring_cnt, -1, [cX, cY])

    #roi = get_ROI_image(image, oi_mask, io_mask)
    #sectors = get_sector_gradients(roi, [cX, cY], inner_ring_cnt, outer_ring_cnt, config["THRESH_SAME_RING"], image, i)
    median_center = [cX, cY] #calculate_center_as_median_intersection_point(sectors, [cX, cY])
    cX, cY = median_center

    output_img_red, output_img_green = get_red_green_masks(black_bg_image)
    output_img_red = remove_small_contours(output_img_red, min_area)
    red_green_image = cv2.add(output_img_red, output_img_green)
    

    inner_ring, outer_ring, inner_ring_cnt, outer_ring_cnt = get_inner_outer_rings(red_green_image)
    #inner_ring_mask, outer_ring_mask= get_ring_segments(im.copy())
    # inner_ring = cv2.merge([inner_ring_full,inner_ring_full,inner_ring_full])
    # outer_ring = cv2.merge([outer_ring_full, outer_ring_full, outer_ring_full])
                
    white_roi_im= subtract_roi(white_bg_image , inner_ring, outer_ring , inner_be ,outer_be ,is_white=True)
    gray_image = preproces_image(white_roi_im)
    start_time = time.time()
    black_bg_image, black_contours, b_heatmap_unnorm, b_heatmap_norm= get_black_contour(gray_image , black_bg_image)

  
    black_roi_im = subtract_roi(black_bg_image,inner_ring, outer_ring, inner_be ,outer_be ,is_white=False )
    gray_image = preproces_image(black_roi_im)
    blurred = cv2.medianBlur(gray_image, 9)
    white_contours, heatmap_unnorm , heatmap_norm= get_white_contour(blurred)
  
    heatmap_unnorm +=b_heatmap_unnorm
    heatmap_norm +=b_heatmap_norm
    # plt.imsave("/app/tmp/heatmap_combined_unnorm_sector.jpg" , heatmap_unnorm, cmap='jet')
    # plt.imsave("/app/tmp/heatmap_combined_norm_sector.jpg" , heatmap_norm, cmap='jet')
        
   
    distance_transform = cv2.distanceTransform(cv2.cvtColor(inner_be, cv2.COLOR_BGR2GRAY), cv2.DIST_L2, 5)
    heatmap_unnorm += distance_transform
    distance_transform = cv2.normalize(distance_transform, None, 0, 1.0, cv2.NORM_MINMAX)
    heatmap_norm += distance_transform
    distance_transform = cv2.distanceTransform(cv2.cvtColor(outer_be, cv2.COLOR_BGR2GRAY), cv2.DIST_L2, 5)
    heatmap_unnorm += distance_transform
    distance_transform = cv2.normalize(distance_transform, None, 0, 1.0, cv2.NORM_MINMAX)
    heatmap_norm += distance_transform    
   
    # plt.imsave("/app/tmp/heatmap_combined_unnorm_be.jpg" , heatmap_unnorm, cmap='jet')
    # plt.imsave("/app/tmp/heatmap_combined_norm_be.jpg" , heatmap_norm, cmap='jet')
    
    gray_image = cv2.cvtColor(output_img_red, cv2.COLOR_BGR2GRAY)
    # Assuming that the image is binary where the contour is white on a black background
    ret, binary = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)
    r_hm_unnorm =  np.zeros_like(binary, dtype = np.float32)
    r_hm_norm =  np.zeros_like(binary, dtype = np.float32)
    
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for idx, c in enumerate(contours[:]):
        mask = np.zeros_like(gray_image)
        cv2.drawContours(mask,  [c], -1, (255), thickness=cv2.FILLED) 
        distance_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        r_hm_unnorm+=distance_transform
        heatmap_normalized = cv2.normalize(distance_transform, None, 0, 1.0, cv2.NORM_MINMAX)
        r_hm_norm+=heatmap_normalized
    heatmap_norm +=r_hm_norm
    #plt.imsave("/app/tmp/heatmap_cring_only.jpg" , r_hm_norm, cmap='jet')
    # plt.imsave("/app/tmp/heatmap_red_only.jpg" , heatmap_normalized, cmap='jet')
    #heatmap_norm = cv2.normalize(heatmap_norm, None, 0, 1.0, cv2.NORM_MINMAX)
    heatmap_unnorm +=r_hm_unnorm
    gray_image = cv2.cvtColor(output_img_green, cv2.COLOR_BGR2GRAY)
    # Assuming that the image is binary where the contour is white on a black background
    ret, binary = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)
    g_hm_unnorm =  np.zeros_like(binary, dtype = np.float32)
    g_hm_norm =  np.zeros_like(binary, dtype = np.float32)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for idx, c in enumerate(contours[:]):
        mask = np.zeros_like(gray_image)
        cv2.drawContours(mask,  [c], -1, (255), thickness=cv2.FILLED) 
        distance_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        g_hm_unnorm+=distance_transform
        heatmap_normalized = cv2.normalize(distance_transform, None, 0, 1.0, cv2.NORM_MINMAX)
        g_hm_norm+=heatmap_normalized
    #heatmap_norm = cv2.normalize(heatmap_norm, None, 0, 1.0, cv2.NORM_MINMAX)
    heatmap_unnorm +=g_hm_unnorm
    heatmap_norm+=g_hm_norm
    test_norm  =  cv2.normalize(g_hm_unnorm, None, 0, 1.0, cv2.NORM_MINMAX)
    # plt.imsave("/app/tmp/heatmap_test_norm.jpg" , test_norm, cmap='jet')
    # plt.imsave("/app/tmp/heatmap_combined_unnorm.jpg" , heatmap_unnorm, cmap='jet')
    # plt.imsave("/app/tmp/heatmap_combined_norm.jpg" , heatmap_norm, cmap='jet')
   
    black_centers = get_contours_centers(black_contours)
    black_bearings ,black_coords = get_bearings(black_centers , cX , cY)
    greatest_angle =max(black_bearings)
    angles=np.array(black_bearings)
    idx= np.argmax(angles)
    if greatest_angle >170 :
        
        black_bearings[idx] = -black_bearings[idx]
        angles[idx] = float('-inf')
        idx =np.argmax(angles)
        if angles[idx] > 170:
            black_bearings[idx] = -black_bearings[idx]
        

    white_centers = get_contours_centers(white_contours)
    white_bearings , white_coords = get_bearings(white_centers , cX ,cY)
    greatest_angle =max(white_bearings)
    angles=np.array(white_bearings)
    idx= np.argmax(angles)
    if greatest_angle >170 :
        white_bearings[idx] = -white_bearings[idx]
        angles[idx] = float('-inf')
        idx =np.argmax(angles)
        if angles[idx] > 170:
            white_bearings[idx] = -white_bearings[idx]

    black_contours , black_bearings, white_contours, white_bearings, black_coords, white_coords= sorts_contour_with_bearings(black_contours , black_bearings, white_contours, white_bearings , black_coords, white_coords)
    scoremap_with_contour = [] 
    for contour, score, angle, black_coord in zip(black_contours, config['BLACK_SECTORS'] , black_bearings , black_coords ):
        obj  = (score, angle, contour , black_coord )
        scoremap_with_contour.append(obj)
    for contour, score, angle , white_coord in zip(white_contours, config['WHITE_SECTORS'] , white_bearings , white_coords ):
        obj  = (score, angle, contour , white_coord)
        scoremap_with_contour.append(obj)
    
    #save_scoremap_with_contour(im , scoremap_with_contour, config)

   
    h,w,c=  image.shape
    pixel_map_all = get_pixel_map(scoremap_with_contour, h, w, inner_be , outer_be, roi_mask)
   
    # plt.imshow(out_im)
    inner_ring_mask , outer_ring_mask = get_ring_mask(black_bg_image.copy() , white_contours, black_contours, inner_be , outer_be,cX,cY)
   
    multiplier_map = get_multiplier_map(inner_ring_mask, outer_ring_mask , h, w)
 
    #multiplier_map = get_multiplier_map(inner_ring_cnt , outer_ring_cnt , h , w)
   # plt.imsave('/app/tmp/multiplier2.jpg' , multiplier_map)
    # #plt.imshow(pixel_map)
    # #print(pixel_map)
    #plt.imsave('/app/tmp/pixel_final2.jpg', pixel_map_all)   
    # #plt.imsave('/app/tmp/pixel_black.jpg', pixel_map_black)   
    #plt.imsave('/app/tmp/pixel_white.jpg', pixel_map_white)   
         
   
    return bullseye_image, inner_ring_mask, outer_ring_mask, inner_be, outer_be,  output_img_red,\
            output_img_green, cX, cY ,pixel_map_all , multiplier_map ,scoremap_with_contour, np.array(heatmap_unnorm,dtype=np.float64) ,np.array(heatmap_norm,dtype=np.float64)

