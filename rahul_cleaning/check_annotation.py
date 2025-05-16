# import cv2
# # from google.colab.patches import cv2_imshow
# # Function to draw bounding boxes on the image
# def draw_bounding_boxes(image_path, annotation_path):
#     # Read image
#     image = cv2.imread(image_path)
#     height, width, _ = image.shape

#     # Open annotation file (YOLO format)
#     with open(annotation_path, 'r') as file:
#         for line in file.readlines():
#             # YOLO format: class_id, x_center, y_center, width, height
#             class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.strip().split())

#             # Convert normalized coordinates to image coordinates
#             x_center = int(x_center * width)
#             y_center = int(y_center * height)
#             bbox_width = int(bbox_width * width)
#             bbox_height = int(bbox_height * height)

#             # Calculate top-left and bottom-right points
#             top_left_x = int(x_center - bbox_width / 2)
#             top_left_y = int(y_center - bbox_height / 2)
#             bottom_right_x = int(x_center + bbox_width / 2)
#             bottom_right_y = int(y_center + bbox_height / 2)

#             # Draw bounding box on the image
#             cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)

#     # Show the image with bounding boxes
#     cv2.imshow(image)
#     cv2.waitKey(0)
#     # cv2.destroyAllWindows()

# # Example usage
# image_path = r'imageCam000_05_11_aug_1.bmp'
# annotation_path = r'imageCam000_05_11_aug_1.txt'
# draw_bounding_boxes(image_path, annotation_path)



import cv2

# Function to draw bounding boxes on the image
def draw_bounding_boxes(image_path, annotation_path):
    # Read image
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Open annotation file (YOLO format)
    with open(annotation_path, 'r') as file:
        for line in file.readlines():
            # YOLO format: class_id, x_center, y_center, width, height
            class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.strip().split())

            # Convert normalized coordinates to image coordinates
            x_center = int(x_center * width)
            y_center = int(y_center * height)
            bbox_width = int(bbox_width * width)
            bbox_height = int(bbox_height * height)

            # Calculate top-left and bottom-right points
            top_left_x = int(x_center - bbox_width / 2)
            top_left_y = int(y_center - bbox_height / 2)
            bottom_right_x = int(x_center + bbox_width / 2)
            bottom_right_y = int(y_center + bbox_height / 2)

            # Draw bounding box on the image
            cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)

    # Show the image with bounding boxes
    cv2.imshow('Image with Bounding Boxes', image)  # Provide a window name
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = r'imageCam000_05_11_aug_1.bmp'
annotation_path = r'imageCam000_05_11_aug_1.txt'
draw_bounding_boxes(image_path, annotation_path)
