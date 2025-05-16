import os
import shutil

# Define the source and destination folder paths
source_folder_path = r"temp_images"
destination_folder_path = r"images_null"

# Ensure the destination folder exists
os.makedirs(destination_folder_path, exist_ok=True)

# Get a list of all files in the source folder
files = os.listdir(source_folder_path)
print(f"Files in source folder: {files}")

# Iterate over each file in the source folder
for file in files:
    file_name, file_ext = os.path.splitext(file)
    print(f"Processing file: {file}, Extension: {file_ext}")
    
    if file_ext.lower() == ".png" or file_ext.lower() == ".jpg" or file_ext.lower() == ".bmp":
        # Check if the corresponding TXT file exists
        txt_file = file_name + ".txt"
        print(f"Looking for corresponding txt file: {txt_file}")
        
        if txt_file not in files:
            # Move the BMP file to the destination folder
            source_file_path = os.path.join(source_folder_path, file)
            destination_file_path = os.path.join(destination_folder_path, file)
            print(f"Moving file {source_file_path} to {destination_file_path}")
            shutil.move(source_file_path, destination_file_path)
        else:
            print(f"Corresponding txt file found for {file}, not moving")
    else:
        print(f"File {file} is not a BMP file, skipping")

print("done")
