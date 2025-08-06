import os
import cv2
import albumentations as A
import numpy as np

# Augmentation count per image
AUG_PER_IMAGE = 5    # You can change to 4 if needed

# Directories
base_dir = "dataset"
image_dir = os.path.join(base_dir, "images")
label_dir = os.path.join(base_dir, "labels")
aug_img_dir = os.path.join(base_dir, "augmented1/images")
aug_lbl_dir = os.path.join(base_dir, "augmented1/labels")

# Create output dirs if they don't exist
os.makedirs(aug_img_dir, exist_ok=True)
os.makedirs(aug_lbl_dir, exist_ok=True)

# Albumentations pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Rotate(limit=45, p=0.5),
    A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=25, val_shift_limit=0, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.0, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=0.25, p=0.5),
    A.RandomGamma(gamma_limit=(90, 110), p=0.5),
    A.GaussianBlur(blur_limit=(1, 3), p=0.5)
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

# Loop through images
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.bmp', '.jpg'))]

for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    label_path = os.path.join(label_dir, os.path.splitext(image_file)[0] + ".txt")

    if not os.path.exists(label_path):
        print(f"No label found for {image_file}, skipping.")
        continue

    image = cv2.imread(image_path)
    orig_height, orig_width = image.shape[:2]

    # Read label and decode polygons
    segments = []
    classes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))
            px_coords = [(coords[i] * orig_width, coords[i+1] * orig_height) for i in range(0, len(coords), 2)]
            segments.append(px_coords)
            classes.append(cls)

    for i in range(AUG_PER_IMAGE):
        # Flatten keypoints
        flat_keypoints = [pt for seg in segments for pt in seg]

        # Apply augmentation
        augmented = transform(image=image, keypoints=flat_keypoints)
        aug_img = augmented['image']
        aug_height, aug_width = aug_img.shape[:2]

        flat_kps = augmented['keypoints']

        # Rebuild segments from flat keypoints
        new_segments = []
        idx = 0
        for seg in segments:
            num_points = len(seg)
            new_seg = flat_kps[idx:idx+num_points]
            idx += num_points
            new_segments.append(new_seg)

        # Save augmented image
        aug_img_name = f"{os.path.splitext(image_file)[0]}_aug_{i}.bmp"
        cv2.imwrite(os.path.join(aug_img_dir, aug_img_name), aug_img)

        # Save augmented label
        aug_lbl_name = f"{os.path.splitext(image_file)[0]}_aug_{i}.txt"
        with open(os.path.join(aug_lbl_dir, aug_lbl_name), "w") as out:
            for cls, seg in zip(classes, new_segments):
                norm_coords = []
                for x, y in seg:
                    norm_x = x / aug_width
                    norm_y = y / aug_height
                    norm_coords.extend([norm_x, norm_y])
                out.write(f"{cls} " + " ".join([f"{c:.6f}" for c in norm_coords]) + "\n")
