import os
import cv2
import numpy as np

# Function to flip bounding box coordinates horizontally
def flip_bbox_horizontally(bbox, img_width):
    x_center, y_center, width, height = bbox
    new_x_center = img_width - x_center
    return [new_x_center, y_center, width, height]

# Function to rotate bounding box coordinates
def rotate_bbox(bbox, angle, img_width, img_height):
    x_center, y_center, width, height = bbox
    angle_rad = np.deg2rad(angle)
    
    # Calculate rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((img_width / 2, img_height / 2), angle, 1)
    rotation_matrix[0, 2] -= img_width / 2
    rotation_matrix[1, 2] -= img_height / 2
    
    # Rotate the center point of the bounding box
    new_center = np.dot(rotation_matrix[:, :2], np.array([x_center - img_width / 2, y_center - img_height / 2])) + [img_width / 2, img_height / 2]
    new_x_center, new_y_center = new_center
    
    return [new_x_center, new_y_center, width, height]

# Function to crop bounding box coordinates
def crop_bbox(bbox, crop_x, crop_y, crop_width, crop_height):
    x_center, y_center, width, height = bbox
    x_center = max(crop_x, min(crop_x + crop_width, x_center))
    y_center = max(crop_y, min(crop_y + crop_height, y_center))
    return [x_center - crop_x, y_center - crop_y, width, height]

# Function to apply all augmentations
def augment_image_and_bbox(img, bbox, img_height, img_width, angle=0, crop_box=None, flip_horizontal=False):
    if img is None:
        print("Error: Input image is None")
        return None, bbox
    
    if angle != 0:
        rotation_matrix = cv2.getRotationMatrix2D((img_width / 2, img_height / 2), angle, 1)
        img = cv2.warpAffine(img, rotation_matrix, (img_width, img_height), borderMode=cv2.BORDER_REFLECT_101)
        bbox = rotate_bbox(bbox, angle, img_width, img_height)
    
    if crop_box is not None:
        crop_x, crop_y, crop_width, crop_height = crop_box
        if crop_x >= img_width or crop_y >= img_height or crop_width <= 0 or crop_height <= 0:
            print(f"Invalid crop box: ({crop_x}, {crop_y}, {crop_width}, {crop_height})")
            return None, bbox
        
        img = img[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
        if img.size == 0:
            print("Error: Cropped image has zero size")
            return None, bbox
        
        img_height, img_width = img.shape[:2]
        bbox = crop_bbox(bbox, crop_x, crop_y, crop_width, crop_height)
    
    if flip_horizontal:
        img = cv2.flip(img, 1)
        bbox = flip_bbox_horizontally(bbox, img_width)
    
    return img, bbox

# Function to process images with augmentations
def process_images_with_augmentations(image_dir, label_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.bmp')]
    
    for image_file in image_files:
        img_path = os.path.join(image_dir, image_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error loading image: {img_path}")
            continue
        
        img_height, img_width = img.shape[:2]
        label_file = image_file.replace('.bmp', '.txt')
        label_path = os.path.join(label_dir, label_file)
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for i in range(5):  # Create 5 augmentations per image
            angle = np.random.choice([0, 90, 180, 270])
            flip_horizontal = np.random.choice([True, False])
            crop_x = np.random.randint(0, img_width // 2)
            crop_y = np.random.randint(0, img_height // 2)
            crop_width = np.random.randint(img_width // 2, img_width)
            crop_height = np.random.randint(img_height // 2, img_height)
            crop_box = (crop_x, crop_y, crop_width, crop_height)
            
            aug_img, aug_bboxes = img.copy(), []
            
            for line in lines:
                data = line.strip().split()
                class_id = data[0]
                bbox = list(map(float, data[1:]))
                aug_img, aug_bbox = augment_image_and_bbox(aug_img, bbox, img_height, img_width, angle, crop_box, flip_horizontal)
                
                if aug_img is None or aug_img.size == 0:
                    print(f"Warning: Image is empty after augmentation for {image_file}")
                    continue
                
                aug_bboxes.append(' '.join([class_id] + list(map(str, aug_bbox))))
            
            # Save augmented image only if it's not empty
            if aug_img is not None and aug_img.size > 0:
                aug_img_path = os.path.join(output_dir, f"augmented_{i}_{image_file}")
                if cv2.imwrite(aug_img_path, aug_img):
                    print(f"Image saved successfully: {aug_img_path}")
                else:
                    print(f"Failed to save image: {aug_img_path}")
            else:
                print(f"Skipping empty image: {image_file}")
            
            # Save augmented label file
            if aug_bboxes:
                aug_label_path = os.path.join(output_dir, f"augmented_{i}_{label_file}")
                with open(aug_label_path, 'w') as f:
                    for bbox_line in aug_bboxes:
                        f.write(f"{bbox_line}\n")

# Example usage
image_dir = "images"
label_dir = "labels"
output_dir = "augmented_data"
process_images_with_augmentations(image_dir, label_dir, output_dir)
