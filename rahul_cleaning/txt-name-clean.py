import os

folder_path = r"frame_data.v4i.yolov5pytorch\train\labels"  # Replace this with the path to your folder
image_extensions = ["png", "jpg", "jpeg", "bmp", "gif"]

# Iterate over files in the folder
for filename in os.listdir(folder_path):
    # Check if the file ends with ".txt" and contains any image extension
    if filename.endswith(".txt") and any(ext in filename for ext in image_extensions):
        # Find the image extension in the filename
        for ext in image_extensions:
            if f"_{ext}" in filename:
                parts = filename.split(f"_{ext}")
                new_filename = parts[0] + ".txt"
                break

        # Check if the new filename already exists
        if new_filename in os.listdir(folder_path):
            print(f"File {new_filename} already exists, skipping...")
            continue

        # Rename the file
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
        print(f"Renamed {filename} to {new_filename}")
