import os
from inference import get_model
import supervision as sv
import cv2

# Function to convert detections to YOLO format
def convert_to_yolo_format(detections, image_shape, output_file):
    height, width, _ = image_shape
    with open(output_file, 'w') as f:
        for detection in detections:
            temp = detection[0]
            x1, y1, x2, y2 = temp[0], temp[1], temp[2], temp[3]
            conf = detection[2]
            class_id = 0
            
            # Calculate YOLO format values
            x_center = (x1 + x2) / 2 / width
            y_center = (y1 + y2) / 2 / height
            bbox_width = (x2 - x1) / width
            bbox_height = (y2 - y1) / height
            
            # Write to file in YOLO format for all detections
            f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f} {conf:.6f}\n")

# Define the folder containing images
image_folder = r"images"  # Replace with your image folder path
output_folder = r"annotate"  # Define the output folder for YOLO files

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# List all image files in the folder
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Load a pre-trained YOLOv8 model
model = get_model(
    model_id="license-plate-recognition-rxg4e/4", api_key="xZlFzpd81h8HQGYlMpjh"
)

# Create supervision annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Set the confidence threshold to 60%
confidence_threshold = 60 / 100  # Convert to a value between 0 and 1

# Process each image in the folder
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)

    # Run inference on the image
    results = model.infer(image)[0]

    # Load the results into the supervision Detections API
    detections = sv.Detections.from_inference(results)
    print("detections", detections)

    # Convert detections to YOLO format and save in the output folder
    output_file = os.path.join(output_folder, os.path.splitext(image_file)[0] + ".txt")  # Save in the new folder
    convert_to_yolo_format(detections, image.shape, output_file)

    # Filter detections for annotation based on confidence threshold
    filtered_indices = [i for i, conf in enumerate(detections.confidence) if conf >= confidence_threshold]

    # Create a Detections object from the filtered detections
    filtered_detections = sv.Detections(
        xyxy=detections.xyxy[filtered_indices],
        class_id=detections.class_id[filtered_indices]
    )

    # Annotate the image with our filtered inference results
    annotated_image = bounding_box_annotator.annotate(scene=image, detections=filtered_detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=filtered_detections)

    # Optionally save or display the annotated image
    # Uncomment the line below to save the annotated image
    # cv2.imwrite(os.path.join(image_folder, f"annotated_{image_file}"), annotated_image)

    # Display the annotated image
    # sv.plot_image(annotated_image)
