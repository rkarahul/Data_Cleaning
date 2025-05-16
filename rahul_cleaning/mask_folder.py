import cv2
import numpy as np
import os

# Define input and output folders
input_folder = 'Camera0'  # Replace with your input folder path
output_folder = 'Camera0_mask'  # Replace with your output folder path

# Create the output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get list of all files in the input folder
file_names = os.listdir(input_folder)

# Loop through each file in the input folder
for file_name in file_names:
    # Create full file path
    file_path = os.path.join(input_folder, file_name)
    
    # Read the image
    image = cv2.imread(file_path)
    if image is None:
        print(f"Skipping file {file_name} (not an image or unable to read)")
        continue

    # Get dynamic coordinates for ROI
    x = 369
    y = 32
    w = 659
    h = 861

    # Create a black mask of the same size as the original image
    full_mask = np.zeros(image.shape[:2], dtype="uint8")

    # Draw a white filled rectangle on the mask for the selected ROI
    cv2.rectangle(full_mask, (x, y), (x + w, y + h), 255, -1)

    # Apply the full mask to the image using bitwise_and
    masked_image_full = cv2.bitwise_and(image, image, mask=full_mask)

    # Define output file path
    output_path = os.path.join(output_folder, file_name)

    # Save the masked image to the specified folder
    cv2.imwrite(output_path, masked_image_full)

    print(f"Masked image saved at: {output_path}")

print("Processing complete.")
