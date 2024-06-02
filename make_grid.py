import cv2
import os
import numpy as np 
import matplotlib.pyplot as plt

# Function to resize image while maintaining aspect ratio
def resize_with_aspect_ratio(image, target_width):
    aspect_ratio = image.shape[1] / image.shape[0]
    target_height = int(target_width / aspect_ratio)
    resized_image = cv2.resize(image, (target_width, target_height))
    return resized_image
if __name__ =='__main__':
        
    # Folder containing images
    parent_folder_path = "D:/dart/singularity_codebase/singularity_runner/images_test_cases_2024_04_25_failure/"
    exps = os.listdir(parent_folder_path)
    for exp in exps: 
        folder_path  = os.path.join(parent_folder_path , exp , "throw") 
     
        # List all image files in the folder
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        print(image_files)
        # Create a grid layout
        grid_width = 2
        grid_height = 2

        # Create a blank canvas for the grid
        grid_canvas_height = 0
        grid_canvas_width = 0

        # Calculate grid cell size based on the first image
        first_image = cv2.imread(os.path.join(folder_path, image_files[0]))
        grid_cell_height, grid_cell_width, _ = first_image.shape

        # Create a grid canvas with appropriate dimensions
        grid_canvas_width = grid_cell_width * grid_width
        grid_canvas_height = grid_cell_height * grid_height
        grid_canvas = np.zeros((grid_canvas_height, grid_canvas_width, 3), dtype=np.uint8)

        # Loop through images and add them to the grid canvas
        for i, image_file in enumerate(image_files):
            # Read the image
            image = cv2.imread(os.path.join(folder_path, image_file))
            
            # Resize image while maintaining aspect ratio
            resized_image = resize_with_aspect_ratio(image, grid_cell_width)
            
            # Calculate position to place the image in the grid
            row_index = i // grid_width
            col_index = i % grid_width
            y_start = row_index * grid_cell_height
            y_end = (row_index + 1) * grid_cell_height
            x_start = col_index * grid_cell_width
            x_end = (col_index + 1) * grid_cell_width
            
            # Add the resized image to the grid canvas
            grid_canvas[y_start:y_end, x_start:x_end] = resized_image

        # Display the grid canvas with all images
        cv2.imwrite( f"D:/dart/singularity_codebase/singularity_runner/grid_outputs/{exp}.jpg",grid_canvas)

        # cv2.imshow("Grid Canvas", grid_canvas)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
