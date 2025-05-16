import os

def remove_unwanted_labels(folder_path):
    """
    Removes label files that do not have a corresponding image file in the same folder.

    Args:
        folder_path (str): Path to the folder containing both image and label files.
    """
    # Supported image file extensions
    image_extensions = {".bmp", ".jpg", ".jpeg", ".png"}

    # Get all files in the folder
    all_files = os.listdir(folder_path)

    # Separate image and label files
    label_files = [f for f in all_files if f.endswith(".txt")]

    for label_file in label_files:
        label_name, _ = os.path.splitext(label_file)

        # Check for corresponding image file
        has_image = any(
            os.path.exists(os.path.join(folder_path, label_name + ext))
            for ext in image_extensions
        )

        # Delete the label file if no corresponding image exists
        if not has_image:
            label_file_path = os.path.join(folder_path, label_file)
            os.remove(label_file_path)
            print(f"Deleted label file: {label_file_path}")


folder_path = r"suffle"  # Replace with your folder path
remove_unwanted_labels(folder_path)


# import os

# def remove_unwanted_labels_and_images(folder_path):
#     """
#     Removes label files and corresponding image files if they do not match.

#     Args:
#         folder_path (str): Path to the folder containing both image and label files.
#     """
#     # Supported image file extensions
#     image_extensions = {".bmp", ".jpg", ".jpeg", ".png"}

#     # Get all files in the folder
#     all_files = os.listdir(folder_path)

#     # Separate label files (with .txt extension)
#     label_files = [f for f in all_files if f.endswith(".txt")]

#     for label_file in label_files:
#         label_name, _ = os.path.splitext(label_file)

#         # Check for corresponding image file (match by name without extension)
#         corresponding_image = None
#         for ext in image_extensions:
#             image_file = label_name + ext
#             if image_file in all_files:
#                 corresponding_image = image_file
#                 break
        
#         # If no matching image file exists, delete both the label file and any image file that might exist
#         if corresponding_image is None:
#             # Delete label file
#             label_file_path = os.path.join(folder_path, label_file)
#             os.remove(label_file_path)
#             print(f"Deleted label file: {label_file_path}")

#             # Now delete any image files that match the label name (if they exist)
#             for ext in image_extensions:
#                 image_file_path = os.path.join(folder_path, label_name + ext)
#                 if os.path.exists(image_file_path):
#                     os.remove(image_file_path)
#                     print(f"Deleted image file: {image_file_path}")

# folder_path = r"All_data_patato"  # Replace with your folder path
# remove_unwanted_labels_and_images(folder_path)
