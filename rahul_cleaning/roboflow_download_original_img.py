# import os
# import requests
# from roboflow import Roboflow

# # Initialize Roboflow with your API key
# rf = Roboflow("0FopmfZXsqPg4hwdfjy7")

# # Connect to the specific project
# project = rf.project("allround_data")

# # List to store image metadata
# records = []

# # Fetch image metadata from the project
# for page in project.search_all(
#     offset=0,
#     limit=100,
#     in_dataset=True,
#     batch=False,
#     fields=["id", "name", "owner"],
# ):
#     records.extend(page)

# print(f"{len(records)} images found")

# # Create a directory to store downloaded images
# os.makedirs("temp_images", exist_ok=True)

# # Download each image
# for record in records:
#     base_url = "https://source.roboflow.com"
#     url = f"{base_url}/{record['owner']}/{record['id']}/original.jpg"

#     try:
#         # Make a GET request to download the image
#         response = requests.get(url)
#         response.raise_for_status()

#         # Save the image locally
#         save_path = os.path.join("temp_images", record["name"])
#         with open(save_path, "wb") as f:
#             f.write(response.content)

#         print(f"Downloaded: {record['name']}")

#     except requests.exceptions.RequestException as e:
#         print(f"Error downloading image: {e}")


import os
import requests
from roboflow import Roboflow

# Initialize Roboflow with your API key
rf = Roboflow("FFfyzuIDUftQ3BZNhrhX")

# Connect to the specific project
project = rf.project("frame_data")

# List to store image metadata
records = []

# Fetch image metadata from the project
for page in project.search_all(
    offset=0,
    limit=100,
    in_dataset=True,
    batch=False,
    fields=["id", "name", "owner"],
):
    records.extend(page)

print(f"{len(records)} images found")

# Create a directory to store downloaded images
os.makedirs("temp_images", exist_ok=True)

# Download each image if not already downloaded
for record in records:
    save_path = os.path.join("temp_images", record["name"])

    if os.path.exists(save_path):
        print(f"Skipped (already exists): {record['name']}")
        continue  # Skip download

    base_url = "https://source.roboflow.com"
    url = f"{base_url}/{record['owner']}/{record['id']}/original.jpg"

    try:
        # Make a GET request to download the image
        response = requests.get(url)
        response.raise_for_status()

        # Save the image locally
        with open(save_path, "wb") as f:
            f.write(response.content)

        print(f"Downloaded: {record['name']}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {record['name']}: {e}")
