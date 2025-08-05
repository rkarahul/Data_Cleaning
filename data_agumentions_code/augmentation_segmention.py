import os
import cv2
import albumentations as A
import numpy as np

# Augmentation count per image
AUG_PER_IMAGE = 4      # Change to 4 if needed

# Directories
base_dir = "valid"
image_dir = os.path.join(base_dir, "images")
label_dir = os.path.join(base_dir, "labels")
aug_img_dir = os.path.join(base_dir, "augmented/images")
aug_lbl_dir = os.path.join(base_dir, "augmented/labels")

# Create output dirs if they don't exist
os.makedirs(aug_img_dir, exist_ok=True)
os.makedirs(aug_lbl_dir, exist_ok=True)

# Albumentations pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.2),
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.09, contrast_limit=0.09, p=0.1), #5% change 
    A.GaussianBlur(blur_limit=(3, 5), p=0.5),  # Random kernel size of 3×3 to 5×5
    A.Rotate(limit=15, p=0.1),
    #A.Blur(p=0.3)
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

# Loop through images
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.bmp','.jpg'))]
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    label_path = os.path.join(label_dir, os.path.splitext(image_file)[0] + ".txt")

    if not os.path.exists(label_path):
        print(f" No label found for {image_file}, skipping.")
        continue

    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Read label and decode polygons
    segments = []
    classes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))
            px_coords = [(coords[i] * width, coords[i+1] * height) for i in range(0, len(coords), 2)]
            segments.append(px_coords)
            classes.append(cls)

    for i in range(AUG_PER_IMAGE):
        flat_keypoints = [pt for seg in segments for pt in seg]
        augmented = transform(image=image, keypoints=flat_keypoints)
        flat_kps = augmented['keypoints']

        # Rebuild segments
        new_segments = []
        idx = 0
        for seg in segments:
            num_points = len(seg)
            new_seg = flat_kps[idx:idx+num_points]
            idx += num_points
            new_segments.append(new_seg)

        # Save image
        aug_img_name = f"{os.path.splitext(image_file)[0]}_aug_{i}.png"
        cv2.imwrite(os.path.join(aug_img_dir, aug_img_name), augmented['image'])

        # Save label
        aug_lbl_name = f"{os.path.splitext(image_file)[0]}_aug_{i}.txt"
        with open(os.path.join(aug_lbl_dir, aug_lbl_name), "w") as out:
            for cls, seg in zip(classes, new_segments):
                norm_coords = []
                for x, y in seg:
                    norm_x = x / width
                    norm_y = y / height
                    norm_coords.extend([norm_x, norm_y])
                out.write(f"{cls} " + " ".join([f"{c:.6f}" for c in norm_coords]) + "\n")
