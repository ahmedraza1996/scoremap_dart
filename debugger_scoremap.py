import csv
import os
import requests
import pandas as pd
import json
import cv2
import numpy as np
from autoscorer.autoscorer import AutoScorer
import matplotlib.pyplot as plt
# Function to create directory if it doesn't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to add text to the image
def add_text_to_image(image_path, text):
    # Read the image
    image = cv2.imread(image_path)
    
    # Define font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_color = (0, 255, 255)  # Bright yellow color (BGR format)
    box_color = (0, 0, 0)  # White color (BGR format)
    
    # Split the text into multiple lines
    text_lines = text.split('\n')
    
    # Calculate text size for each line and find the maximum width and total height
    max_width = 0
    total_height = 0
    text_sizes = []
    for line in text_lines:
        text_size, _ = cv2.getTextSize(line, font, font_scale, font_thickness)
        text_sizes.append(text_size)
        max_width = max(max_width, text_size[0])
        total_height += text_size[1]
    
    # Calculate text position (bottom right corner with some padding)
    text_x = image.shape[1] - max_width - 10
    text_y = image.shape[0] - 20  # Start at the bottom edge with some padding
    
    # Draw a white box behind the text
    box_top_left = (text_x - 5, text_y - total_height - 15)
    box_bottom_right = (text_x + max_width + 5, text_y + 5)
    cv2.rectangle(image, box_top_left, box_bottom_right, box_color, cv2.FILLED)
    
    # Put text on the image
    for line, text_size in zip(text_lines[::-1], text_sizes[::-1]):
        # Adjust text position relative to the box
        text_baseline = text_y - 5
        text_y -= text_size[1] + 5  # Subtract text height and add spacing between lines
        cv2.putText(image, line, (text_x, text_baseline), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    return image


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


def process_throws(cam_number, images_dir, exp_name, board_id ,  desired_output):
    board_path= f"output/boardmap_json/{board_id}.json"
    import pickle
   
    with open(board_path, "rb") as json_file:
        # Load the data from the JSON file
        data = pickle.load(json_file)

    url = row[f'Cam {cam_number} Link']
    response = requests.get(url)
    if response.status_code == 200:
        filename = url.split('/')[-1]
        with open(os.path.join(images_dir, exp_name, 'throw', f'{cam_number}.jpg'), 'wb') as f:
             f.write(response.content)
        image_path = os.path.join(images_dir, exp_name, 'throw', f'{cam_number}.jpg')
        cam_records = [record for record in desired_output if record.get('Cam') == cam_number]
        
        text = ""
        for record in cam_records:
            text += str(record) + "\n"
        image_with_text = add_text_to_image(image_path, text)
        
        cam_data = [(entry['dx'], entry['dy']) for entry in desired_output if entry['Cam'] == cam_number]
        for dx, dy in cam_data:
            dx = int(dx)
            dy = int(dy)
            cv2.circle(image_with_text, (dx, dy), circle_radius, circle_color, -1)
        
        # heeree logic of board to throw
       
        plt.imsave(os.path.join(images_dir, exp_name, 'throw',f"pixel_map_{cam_number}.jpg") , data[cam_number]['PIXEL_MAP']*10)

        pixel_map = cv2.imread(os.path.join(images_dir, exp_name, 'throw',f"pixel_map_{cam_number}.jpg"))
        overlayed_image = cv2.addWeighted(image_with_text,1,pixel_map, 0.8, 0)
        plt.imsave(os.path.join(images_dir, exp_name, 'throw',f"multiplier_map_{cam_number}.jpg") , data[cam_number]['MULTIPLIER_MAP'])

        pixel_map = cv2.imread(os.path.join(images_dir, exp_name, 'throw',f"multiplier_map_{cam_number}.jpg"))
        overlayed_image = cv2.addWeighted(overlayed_image,1,pixel_map, 0.2, 0)
       
        for item in data[cam_number]['SCORE_MAP'][:-1]:
    
            pt_in = item[1]['pt_in']
            pt_out = item[1]['pt_out']
            # Extrapolate the line passing through pt1 and pt2
            extrapolated_pt1, extrapolated_pt2, extrapolated_pt3, extrapolated_pt4 = extrapolate_line(pt_out, pt_in, 190)

            # Plot a line between pt_in and pt_out on the canvas
            cv2.line(overlayed_image, (extrapolated_pt1, extrapolated_pt2), (extrapolated_pt3, extrapolated_pt4), (0, 255, 0), thickness=2)  # Green line, thickness=2

            cv2.imwrite(image_path, overlayed_image)
def process_board(cam_number, images_dir, exp_name, desired_output):
    url= row[f'Board {cam_number} Link']
    response=requests.get(url)
    if response.status_code == 200:
        filename = str(cam_number)+'.jpg'
        image_path = os.path.join(images_dir+'/'+exp_name+'/board', filename)
      
        with open(image_path, 'wb') as f:
            f.write(response.content)
        


if __name__ =='__main__':
    df = pd.read_csv('D:/dart/test_case1.csv')
    # df_cleaned = df.dropna(how='all')
    # filtered_df = df_cleaned[df_cleaned['Is Correct'] == 'No']
    # unique_calibration_ids = filtered_df['Calibration ID'].unique()
    # df_cleaned = df_cleaned[df_cleaned['Calibration ID'].isin(unique_calibration_ids)]
    # throw_ids = df_cleaned['Throw ID'].unique()
    # throw_df=df_cleaned[df_cleaned['Throw ID'].isin(throw_ids)]
    # throw_df =throw_df[['Throw ID',  'Is Correct' , 'Detected Value', 'Correct Value' ,'Previous State']]
    # #th_df =pd.DataFrame(throw_ids)
    # file_path = 'my_list_2.csv'

    # # Save the DataFrame to a CSV file
    # throw_df.to_csv(file_path, index=False, header=False)




    images_dir = 'images_test_cases_2024_05_03_failure_sing'
    import threading
    import requests
    import os
    import cv2
    circle_color=(0, 255, 255) 
    circle_radius=2
# Define parameters
    cameras = [0, 1, 2, 3]

    for index, row in df.iterrows():
        print(row['Throw ID'])
        exp_name = 'b_'+str(row['Calibration ID'])+'t_'+str(row['Throw ID'])
        
        if not os.path.exists(images_dir+'/'+exp_name+'/throw'):
            os.makedirs(images_dir+'/'+exp_name+'/throw')
        if not os.path.exists(images_dir+'/'+exp_name+'/board'):
            os.makedirs(images_dir+'/'+exp_name+'/board')
        json_data = json.loads(row['Raw output'])
        
        desired_output = [entry for entry in json_data if entry.get("Cam") is not None and entry.get("darts") is not None]
        # Create threads for each camera
        board_images_paths =[]
        threads = []
        for cam_number in cameras:
            thread = threading.Thread(target=process_board, args=(cam_number, images_dir, exp_name, desired_output))
            board_images_paths.append(os.path.join(images_dir+'/'+exp_name+'/board', str(cam_number)+'.jpg'))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to finish
        for thread in threads:
            thread.join()
        
        
        instance = AutoScorer(str(row['Calibration ID']), board_images_paths)

        instance.calculate_board_map_and_cache()
        
        threads = []
        for cam_number in cameras:
            thread = threading.Thread(target=process_throws, args=(cam_number, images_dir, exp_name,str(row['Calibration ID']), desired_output))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to finish
        for thread in threads:
            thread.join()
        

        # All images processed
    print("All images processed successfully.")
