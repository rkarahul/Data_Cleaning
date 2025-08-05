import os
import cv2
import numpy as np

# Paths
aug_img_dir = r"valid/augmented/images"
aug_lbl_dir = r"valid/augmented/labels"
visual_dir = r"valid/augmented/visualization"

# Create output directory
os.makedirs(visual_dir, exist_ok=True)

# Load all augmented images
aug_images = [f for f in os.listdir(aug_img_dir) if f.endswith(('.png', '.bmp'))]

for image_file in aug_images:
    image_path = os.path.join(aug_img_dir, image_file)
    label_path = os.path.join(aug_lbl_dir, os.path.splitext(image_file)[0] + ".txt")

    if not os.path.exists(label_path):
        print(f"[!] Missing label for {image_file}, skipping.")
        continue

    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            cls_id = int(parts[0])
            coords = list(map(float, parts[1:]))

            # Convert normalized coords to pixel coords
            points = [(int(x * width), int(y * height)) for x, y in zip(coords[::2], coords[1::2])]
            pts_array = np.array(points, np.int32).reshape((-1, 1, 2))

            # Color by class (extend if more classes)
            color = (0, 0, 255) if cls_id == 0 else (255, 0, 0)

            # Draw polygon
            cv2.polylines(image, [pts_array], isClosed=True, color=color, thickness=2)

            # Optionally draw class text at the first point
            cv2.putText(image, str(cls_id), points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    # Save visualization
    vis_path = os.path.join(visual_dir, image_file)
    cv2.imwrite(vis_path, image)
    print(f"[✓] Visual saved: {vis_path}")
