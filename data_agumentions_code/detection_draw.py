import os
import cv2

# Define folder paths
image_folder = 'aug_images\images'
label_folder = 'aug_images\labels'
output_folder = 'aug_images\output'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# List all image files
image_extensions = ('.jpg', '.jpeg', '.png','.bmp')
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(image_extensions)]

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    label_file = os.path.splitext(image_file)[0] + '.txt'
    label_path = os.path.join(label_folder, label_file)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        continue

    image_height, image_width = image.shape[:2]

    # Check if label file exists
    if not os.path.isfile(label_path):
        print(f"No label file for: {image_file}")
        continue

    # Read annotation file
    with open(label_path, 'r') as f:
        lines = f.readlines()

    # Draw boxes
    for line in lines:
        if line.strip() == "":
            continue
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        class_id, x_center, y_center, width, height = map(float, parts)

        # Denormalize coordinates
        x_center *= image_width
        y_center *= image_height
        width *= image_width
        height *= image_height

        # Convert to top-left and bottom-right points
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Save annotated image
    output_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_path, image)
    print(f"Saved annotated image: {output_path}")
