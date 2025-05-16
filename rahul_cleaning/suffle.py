# import os
# import uuid

# # Define the directory where images and text files are stored together
# folder_dir = 'data'

# # Allowed image file extensions (you can modify this list if needed)
# image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

# def rename_image_and_txt(directory):
#     # Loop through files in the directory
#     for filename in os.listdir(directory):
#         # Get file extension
#         file_extension = os.path.splitext(filename)[1]
#         file_name_without_ext = os.path.splitext(filename)[0]

#         # Check if the file is an image
#         if file_extension.lower() in image_extensions:
#             # Check if a corresponding .txt file exists
#             txt_file = file_name_without_ext + '.txt'
#             if os.path.exists(os.path.join(directory, txt_file)):
#                 # Generate a random name
#                 new_name = str(uuid.uuid4())

#                 # Rename the image file
#                 old_image = os.path.join(directory, filename)
#                 new_image = os.path.join(directory, new_name + file_extension)
#                 os.rename(old_image, new_image)

#                 # Rename the corresponding text file
#                 old_txt = os.path.join(directory, txt_file)
#                 new_txt = os.path.join(directory, new_name + '.txt')
#                 os.rename(old_txt, new_txt)

#                 print(f"Renamed: {filename} and {txt_file} -> {new_name}{file_extension} and {new_name}.txt")

# def rename_images_and_texts_in_folder():
#     print("Renaming images and corresponding text files...")
#     # Rename images and their corresponding text files
#     rename_image_and_txt(folder_dir)

# if __name__ == "__main__":
#     rename_images_and_texts_in_folder()


# random shuffle

import os
import random
import shutil

def shuffle_and_save_data(source_folder, destination_folder):
   # Create the destination folder if it doesn't exist
   if not os.path.exists(destination_folder):
       os.makedirs(destination_folder)

   # Get a list of all files in the source folder
   file_list = os.listdir(source_folder)
   print("starting")
   # Shuffle the list of files
   random.shuffle(file_list)

   for i, filename in enumerate(file_list):
       # Check if it is an image file
       if filename.endswith('.bmp') or filename.endswith('.jpg') or filename.endswith('.png'):
           # Get the image file name and corresponding label file name
           image_file_path = os.path.join(source_folder, filename)
           label_file_path = os.path.join(source_folder, filename.replace('.jpg', '.txt').replace('.bmp', '.txt').replace('.png', '.txt'))

           # Check if the corresponding label file exists
           if os.path.exists(label_file_path):
               # Rename the image and label files with new numeric names
               new_image_name = str(i + 1) + '.bmp'
               new_label_name = str(i + 1) + '.txt'

               # Copy the image and label files to the destination folder with the new names
               shutil.copy(image_file_path, os.path.join(destination_folder, new_image_name))
               shutil.copy(label_file_path, os.path.join(destination_folder, new_label_name))

if __name__ == "__main__":
   # Set the source and destination folder paths
   source_folder = r"temp_images\images"  
   target_folder = r"suffle"

   shuffle_and_save_data(source_folder, target_folder)
