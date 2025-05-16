import os
from inference import get_model
import supervision as sv
import cv2

# Function to convert detections to YOLO format
def convert_to_yolo_format(detections, image_shape, output_file):
    height, width, _ = image_shape
    with open(output_file, 'w') as f:
        for detection in detections:
            # Each detection format: [x1, y1, x2, y2, confidence, class_id]
            temp = detection[0]
            x1, y1, x2, y2 = temp[0], temp[1], temp[2], temp[3]
            conf = detection[2]
            class_id = 0  # Set class_id to 0 for license plates
            
            # Calculate YOLO format values
            x_center = (x1 + x2) / 2 / width
            y_center = (y1 + y2) / 2 / height
            bbox_width = (x2 - x1) / width
            bbox_height = (y2 - y1) / height
            
            # Write to file in YOLO format
            f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

# Function to process a folder of images and save detections in YOLO format
def process_image_folder(input_folder, output_folder, model):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Iterate over all images in the input folder
    for image_file in os.listdir(input_folder):
        if image_file.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(input_folder, image_file)
            image = cv2.imread(image_path)
            
            # Run inference
            results = model.infer(image)[0]
            
            # Load the results into supervision Detections API
            detections = sv.Detections.from_inference(results)
            
            # Generate output file path
            output_file = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}.txt")
            
            # Convert detections to YOLO format and save to output folder
            convert_to_yolo_format(detections, image.shape, output_file)
            
            # Annotate and optionally display the image
            bounding_box_annotator = sv.BoundingBoxAnnotator()
            label_annotator = sv.LabelAnnotator()
            annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
            
            # Display the annotated image (optional)
            # sv.plot_image(annotated_image)  # Uncomment to display each annotated image

# Main part of the script
input_folder = r"Data"  # Input folder containing images
output_folder = r"Data_output"  # Output folder to save YOLO format txt files

# Load the pre-trained YOLOv8 model
model = get_model(model_id="wheels-detection-vuaey/1", api_key="O9wrha5Ovu1JB6f7qavk")

# Process the image folder and save YOLO format txt files
process_image_folder(input_folder, output_folder, model)
