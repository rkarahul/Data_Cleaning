import os
import shutil
from collections import defaultdict

def move_files_based_on_classes(dataset_dir, image_dir, destination_dir, target_classes):
    """
    Moves label and image files from the dataset to the destination folder
    if any of the target classes are present in the labels.

    Args:
        dataset_dir (str): Directory containing the label (.txt) files.
        image_dir (str): Directory containing the image files.
        destination_dir (str): Destination directory where matched files are moved.
        target_classes (set): Set of target class labels to filter.
    
    Returns:
        dict: Dictionary of class counts.
        int: Count of blank files deleted.
    """
    # Ensure destination directory exists
    os.makedirs(destination_dir, exist_ok=True)

    # Dictionary to store class frequencies and blank file count
    class_counts = defaultdict(int)
    blank_file_count = 0

    # Iterate through all label files in the dataset directory
    for file_name in os.listdir(dataset_dir):
        if file_name.endswith(".txt"):
            with open(os.path.join(dataset_dir, file_name), 'r') as f:
                lines = f.readlines()

            # Skip and delete blank files
            if len(lines) == 0:
                os.remove(os.path.join(dataset_dir, file_name))
                blank_file_count += 1
                continue

            # Extract class labels from the file
            classes = [line.split()[0] for line in lines]

            # Check if any target class exists in the file
            if any(cls in target_classes for cls in classes):
                shutil.move(os.path.join(dataset_dir, file_name), os.path.join(destination_dir, file_name))

                # Move corresponding image file if found
                image_base_name = os.path.splitext(file_name)[0]
                for ext in ['.jpg', '.jpeg', '.png']:  # Check multiple image extensions
                    image_file = os.path.join(image_dir, image_base_name + ext)
                    if os.path.exists(image_file):
                        shutil.move(image_file, os.path.join(destination_dir, os.path.basename(image_file)))
                        break
            else:
                # Increment class count if no target class found
                for class_label in classes:
                    class_counts[class_label] += 1

    return class_counts, blank_file_count
