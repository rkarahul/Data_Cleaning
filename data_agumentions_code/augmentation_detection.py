"""
import os
import cv2
import numpy as np
import albumentations as A
from glob import glob

# Paths
images_path = r'G:\apple\Apple_Project_Finall.v1i.yolov8\train\images'  # Folder containing images
annotations_path = r'G:\apple\Apple_Project_Finall.v1i.yolov8\train\labels/'  # Folder containing YOLOv8 annotations
output_images_path = r'G:\apple\Apple_Project_Finall.v1i.yolov8\images'  # Folder to save augmented images
output_annotations_path = r'G:\apple\Apple_Project_Finall.v1i.yolov8\labels'  # Folder to save augmented annotations

# Create output directories if not exist
os.makedirs(output_images_path, exist_ok=True)
os.makedirs(output_annotations_path, exist_ok=True)

# Define augmentations
augmentations = [
    A.RandomBrightnessContrast(p=1.0, brightness_limit=(-0.4, 0.6), contrast_limit=(-0.4, 0.6)),
    A.MotionBlur(blur_limit=(5, 7), p=1.0),
    A.GaussNoise(var_limit=(0.02, 0.02), p=1.0)
]
    #A.HorizontalFlip(p=1.0),
    #A.VerticalFlip(p=1.0),
    #A.Rotate(limit=270, p=1.0),
# Read and apply augmentations
for img_file in glob(os.path.join(images_path, '*.png')):
    img_name = os.path.basename(img_file)
    image = cv2.imread(img_file)
    
    for i, aug in enumerate(augmentations):
        augmented = aug(image=image)
        aug_image = augmented['image']
        
        # Save augmented image
        aug_image_name = f"{os.path.splitext(img_name)[0]}_aug_{i}.jpg"
        cv2.imwrite(os.path.join(output_images_path, aug_image_name), aug_image)
        
        # Copy corresponding annotation
        annotation_file = os.path.join(annotations_path, f"{os.path.splitext(img_name)[0]}.txt")
        aug_annotation_file = os.path.join(output_annotations_path, f"{os.path.splitext(img_name)[0]}_aug_{i}.txt")
        if os.path.exists(annotation_file):
            with open(annotation_file, 'r') as f:
                annotation_data = f.read()
            with open(aug_annotation_file, 'w') as f:
                f.write(annotation_data)

print("Data augmentation completed.")
"""
# import os
# import cv2
# import numpy as np
# import albumentations as A
# from glob import glob

# # Paths
# images_path = r'G:\apple\line2'  # Folder containing images
# annotations_path = r'G:\apple\Apple_Project_Finall.v1i.yolov8\train\labels/'  # Folder containing YOLOv8 annotations
# output_images_path = r'G:\apple\line2\au'  # Folder to save augmented images
# output_annotations_path = r'G:\apple\Apple_Project_Finall.v1i.yolov8\labels'  # Folder to save augmented annotations

# # Create output directories if not exist
# os.makedirs(output_images_path, exist_ok=True)
# os.makedirs(output_annotations_path, exist_ok=True)

# # Define augmentations
# augmentations = [
#     A.RandomBrightnessContrast(p=1.0, brightness_limit=(-0.3, 0.5), contrast_limit=(-0.3, 0.5)),
#     A.MotionBlur(blur_limit=(5, 7), p=1.0),
#     A.GaussNoise(var_limit=(0.02, 0.02), p=1.0)
# ]
#     #A.HorizontalFlip(p=1.0),
#     #A.VerticalFlip(p=1.0),
#     #A.Rotate(limit=270, p=1.0),
# # Read and apply augmentations
# for img_file in glob(os.path.join(images_path, '*.jpg')):
#     img_name = os.path.basename(img_file)
#     image = cv2.imread(img_file)
    
#     for i, aug in enumerate(augmentations):
#         augmented = aug(image=image)
#         aug_image = augmented['image']
        
#         # Save augmented image
#         aug_image_name = f"{os.path.splitext(img_name)[0]}_aug_{i}.jpg"
#         cv2.imwrite(os.path.join(output_images_path, aug_image_name), aug_image)

# print("Data augmentation completed.")
import os
import cv2
import numpy as np
import albumentations as A
from glob import glob

# Paths
images_path = r'data\images'
annotations_path = r'data\labels'
output_images_path = r'aug_images/images'
output_annotations_path = r'aug_images/labels'

# Create output directories
os.makedirs(output_images_path, exist_ok=True)
os.makedirs(output_annotations_path, exist_ok=True)

# Define bbox params for Albumentations
bbox_params = A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3)

# Define augmentations
transformations = [
    A.RandomBrightnessContrast(p=1.0, brightness_limit=(-0.4, 0.6)),
    A.MotionBlur(blur_limit=(5, 7), p=1.0),
    A.GaussNoise(var_limit=(0.02, 0.02), p=1.0),
    A.Compose([A.HorizontalFlip(p=1.0)], bbox_params=bbox_params),
    A.Compose([A.VerticalFlip(p=1.0)], bbox_params=bbox_params),
    A.Compose([A.Rotate(limit=360, border_mode=cv2.BORDER_CONSTANT, p=1.0)], bbox_params=bbox_params)
]

# Process each image and its label
for img_file in glob(os.path.join(images_path, '*.bmp')):
    img_name = os.path.basename(img_file)
    image = cv2.imread(img_file)
    h, w = image.shape[:2]

    # Read corresponding label file
    label_file = os.path.join(annotations_path, os.path.splitext(img_name)[0] + '.txt')
    bboxes = []
    class_labels = []

    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    bbox = list(map(float, parts[1:]))
                    bboxes.append(bbox)
                    class_labels.append(class_id)

    # Skip image if no annotations found
    if not bboxes:
        continue

    # Apply augmentations
    for i, transform in enumerate(transformations):
        try:
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_image = augmented['image']
            aug_bboxes = augmented['bboxes']
            aug_labels = augmented['class_labels']
        except Exception as e:
            print(f"❌ Skipping augmentation {i} for {img_name} due to error: {e}")
            continue

        # Skip if no bboxes survived after augmentation
        if len(aug_bboxes) == 0:
            continue

        # Save augmented image
        aug_img_name = f"{os.path.splitext(img_name)[0]}_aug_{i}.bmp"
        cv2.imwrite(os.path.join(output_images_path, aug_img_name), aug_image)

        # Save updated label file
        aug_label_path = os.path.join(output_annotations_path, os.path.splitext(aug_img_name)[0] + '.txt')
        with open(aug_label_path, 'w') as f:
            for label, bbox in zip(aug_labels, aug_bboxes):
                # Clamp each value to ensure it's within [0.0, 1.0]
                bbox = [max(0.0, min(1.0, x)) for x in bbox]
                f.write(f"{label} {' '.join([f'{x:.6f}' for x in bbox])}\n")

print("✅ Object detection data augmentation completed.")



