# import cv2
# import numpy as np

# # Load the image
# image = cv2.imread('1.bmp')  # Replace 'your_image.jpg' with your image file
# # Create a black mask of the same size as the original image
# full_mask = np.zeros(image.shape[:2], dtype="uint8")
# # roi=image[355:643,34:849]
# # Draw a white filled rectangle on the mask for the selected ROI
# cv2.rectangle(full_mask, (360,80), (1000,880), 255,-1)
# # Apply the full mask to the image using bitwise_and
# masked_image_full = cv2.bitwise_and(image, image, mask=full_mask)

# # Display the masked image
# cv2.imshow("Masked Image", masked_image_full)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# import cv2
# import numpy as np

# # Load the image
# image = cv2.imread('cam2_18_37_23 (1).bmp')  # Replace 'your_image.jpg' with your image file

# # Get dynamic coordinates for ROI
#  -------- cam3  ------------
# x = 387
# y = 124
# w = 560
# h = 863

# -------- cam2 ----

# x = 143
# y = 134
# w = 496
# h = 856


# -- cam1 ------

# x = 352
# y = 75
# w = 631
# h = 879

# ----cam0 ----

# x = 369
# y = 32
# w = 659
# h = 861


# # Create a black mask of the same size as the original image
# full_mask = np.zeros(image.shape[:2], dtype="uint8")

# # Draw a white filled rectangle on the mask for the selected ROI
# cv2.rectangle(full_mask, (x, y), (x + w, y + h), 255, -1)

# # Apply the full mask to the image using bitwise_and
# masked_image_full = cv2.bitwise_and(image, image, mask=full_mask)

# # Display the masked image
# cv2.imshow("Masked Image", masked_image_full)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np

# Load the image
image = cv2.imread('cam0_18_37_18 (1).bmp')  # Replace 'your_image.jpg' with your image file
# Create a black mask of the same size as the original image
full_mask = np.zeros(image.shape[:2], dtype="uint8")
# Select ROI
# select qudinate value roi (337, 48, 697, 820)
roi = cv2.selectROI("Select ROI", image, fromCenter=False, showCrosshair=True)
print("roi",roi)
# Draw a white filled rectangle on the mask for the selected ROI
cv2.rectangle(full_mask, (int(roi[0]), int(roi[1])), (int(roi[0]+roi[2]), int(roi[1]+roi[3])), 255, -1)
# Apply the full mask to the image using bitwise_and
masked_image_full = cv2.bitwise_and(image, image, mask=full_mask)
# Display the original image, the full mask, and the masked image
cv2.imshow("Masked Image", masked_image_full)
cv2.waitKey(0)
cv2.destroyAllWindows()