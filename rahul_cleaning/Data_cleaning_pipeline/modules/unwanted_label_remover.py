import os

def remove_unwanted_labels(image_folder_path, label_folder_path):
    """
    Removes label files that do not have a corresponding image file.

    Args:
        image_folder_path (str): Path to the folder containing image files.
        label_folder_path (str): Path to the folder containing label files.
    """
    # Get list of image and label files
    image_files = os.listdir(image_folder_path)
    label_files = os.listdir(label_folder_path)

    # Iterate through each label file
    for label_file in label_files:
        label_name, label_ext = os.path.splitext(label_file)
        
        if label_ext.lower() == ".txt":
            image_file = label_name + ".jpg"
            image_file_path = os.path.join(image_folder_path, image_file)
            txt_file_path = os.path.join(label_folder_path, label_file)

            # Delete the label file if no corresponding image exists
            if not os.path.exists(image_file_path):
                os.remove(txt_file_path)
