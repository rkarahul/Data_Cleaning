import os
import shutil

def check_image_label_match(source_image_folder_path, label_folder_path, destination_folder_path):
    """
    Moves images to a separate folder if they don't have corresponding label (.txt) files.

    Args:
        source_image_folder_path (str): Path to the folder containing image files.
        label_folder_path (str): Path to the folder containing label files.
        destination_folder_path (str): Path to the folder where unmatched images will be moved.
    """
    # Ensure destination directory exists
    os.makedirs(destination_folder_path, exist_ok=True)
    
    # Get a list of all image files in the source directory
    image_files = os.listdir(source_image_folder_path)

    # Iterate through each image file
    for image_file in image_files:
        file_name, file_ext = os.path.splitext(image_file)
        
        # Check for JPG files
        if file_ext.lower() == ".jpg":
            txt_file = file_name + ".txt"
            txt_file_path = os.path.join(label_folder_path, txt_file)
            
            # If the label file doesn't exist, move the image to the destination folder
            if not os.path.exists(txt_file_path):
                source_image_file_path = os.path.join(source_image_folder_path, image_file)
                destination_file_path = os.path.join(destination_folder_path, image_file)
                shutil.move(source_image_file_path, destination_file_path)
