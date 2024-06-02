from autoscorer.autoscorer import AutoScorer
import numpy as np 
import matplotlib.pyplot as plt

def extrapolate_line(pt1, pt2, length):
    x1, y1 = pt1
    x2, y2 = pt2
    dx = x2 - x1
    dy = y2 - y1
    # Calculate the slope
    if dx != 0:
        m = dy / dx
        # Calculate additional points along the line
        x3 = x2 + length * np.cos(np.arctan(m))
        y3 = y2 + length * np.sin(np.arctan(m))
        x4 = x2 - length * np.cos(np.arctan(m))
        y4 = y2 - length * np.sin(np.arctan(m))
    else:
        x3 = x2
        y3 = y2 + length
        x4 = x2
        y4 = y2 - length
    return int(x3), int(y3), int(x4), int(y4)


if __name__ =='__main__':
    # TODO load board id and images here
    
    board_id = 't1'
    board_images_paths =["D:/dart/2.jpg",
                         "D:/dart/2.jpg",
                         "D:/dart/2.jpg",
                        "D:/dart/2.jpg"]

    instance = AutoScorer(board_id, board_images_paths)

    instance.calculate_board_map_and_cache()

    # import pickle
    # import cv2
    # # Open the JSON file
    # with open("D:/dart/.json", "rb") as json_file:
    #     # Load the data from the JSON file
    #     data = pickle.load(json_file)

    # # Process the loaded data
    # #print(data)  # or perform any other operation on the loaded data
    
    # img= data[0]['IMAGE'].copy()
    
   

    # Display the canvas with the lines
  
    # # Plot the line on the image
    # image_with_line = cv2.line(img.copy(), pt_in, pt_out, (0, 255, 0), thickness=2)  # Green line, thickness=2

    # Display the image with the line
    # cv2.imshow("Image with Line", image_with_line)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    # mask[200:400, 200:400] = 255  # Example: setting a rectangular region to 1
    # mask_3channels = cv2.merge([mask, mask, mask])

    # Convert the mask to 3 channels for overlaying
   
    # overlayed_image = cv2.addWeighted(img,1, data[0]['MULT_2'], 0.6, 1)
    # overlayed_image = cv2.addWeighted(overlayed_image, 1, data[0]['MULT_3'], 0.6, 1)
    # overlayed_image = cv2.addWeighted(overlayed_image,1, data[0]['INNER_BE'], 0.6, 1)
    # overlayed_image = cv2.addWeighted(overlayed_image,1, data[0]['OUTER_BE'], 0.6, 1)
    # overlayed_image = cv2.addWeighted(overlayed_image,1, data[0]['ROI'], 0.6, 1)


    # for item in data[0]['SCORE_MAP'][:-1]:
  
    #     pt_in = item[1]['pt_in']
    #     pt_out = item[1]['pt_out']
    #     # Extrapolate the line passing through pt1 and pt2
    #     extrapolated_pt1, extrapolated_pt2, extrapolated_pt3, extrapolated_pt4 = extrapolate_line(pt_out, pt_in, 190)

    #     # Plot a line between pt_in and pt_out on the canvas
    #     cv2.line(overlayed_image, (extrapolated_pt1, extrapolated_pt2), (extrapolated_pt3, extrapolated_pt4), (0, 255, 0), thickness=2)  # Green line, thickness=2

    # # # Display the image using OpenCV
    # cv2.imshow('Image', overlayed_image)
    # cv2.waitKey(0)  # Wait for a key press (0 means wait indefinitely)
    # cv2.destroyAllWindows()  # Close all OpenCV windows


