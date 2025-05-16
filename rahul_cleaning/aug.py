import os
import cv2
import numpy as np

# Function to flip bounding box coordinates vertically
def flip_bbox_vertically(bbox, img_height):
   x_center, y_center, width, height = bbox
   new_y_center = img_height - y_center
   return [x_center, new_y_center, width, height]

# Function to flip images and corresponding bounding boxes vertically
def flip_images_and_boxes_vertically(image_dir, label_dir, output_dir):
   os.makedirs(output_dir, exist_ok=True)
   
   # List image files
   image_files = [f for f in os.listdir(image_dir) if f.endswith('.bmp')]

   for image_file in image_files:
       # Read image
       img_path = os.path.join(image_dir, image_file)
       img = cv2.imread(img_path)
       img_height = img.shape[0]
       
       # Flip image vertically
       flipped_img = cv2.flip(img, 0)
       
       # Save flipped image
       flipped_img_path = os.path.join(output_dir, f"flipped_{image_file}")
       cv2.imwrite(flipped_img_path, flipped_img)
       
       # Read corresponding label file
       label_file = image_file.replace('.bmp', '.txt')
       label_path = os.path.join(label_dir, label_file)
       
       # Flip bounding box coordinates vertically and save in new label file
       with open(label_path, 'r') as f:
           lines = f.readlines()
       
       flipped_lines = []
       for line in lines:
           data = line.strip().split()
           class_id = data[0]
           bbox = list(map(float, data[1:]))
           flipped_bbox = flip_bbox_vertically(bbox, img_height)
           flipped_line = ' '.join([class_id] + list(map(str, flipped_bbox))) + '\n'
           flipped_lines.append(flipped_line)
       
       flipped_label_path = os.path.join(output_dir, f"flipped_{label_file}")
       with open(flipped_label_path, 'w') as f:
           f.writelines(flipped_lines)
# Example usage
image_dir = r"images"
label_dir = r"binary_class_label"
output_dir = r"flipped_data_vertical"

flip_images_and_boxes_vertically(image_dir, label_dir, output_dir)