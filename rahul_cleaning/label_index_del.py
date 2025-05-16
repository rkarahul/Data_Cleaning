# import os

# def modify_annotations(folder_path):
#     """
#     Modifies the label (.txt) files within a folder by removing lines with specified class IDs.

#     Args:
#         folder_path (str): Path to the folder containing label (.txt) files.
#     """
#     # Class IDs to remove
#     classes_to_remove = {'0', '1', '2', '3', '4', '5', '7', '8'}

#     for filename in os.listdir(folder_path):
#         if filename.endswith(".txt"):
#             file_path = os.path.join(folder_path, filename)

#             # Read the lines in the label file
#             with open(file_path, 'r') as file:
#                 lines = file.readlines()

#             # Rewrite the file without lines starting with any of the specified class IDs
#             with open(file_path, 'w') as file:
#                 for line in lines:
#                     parts = line.strip().split()
#                     if parts and parts[0] not in classes_to_remove:
#                         file.write(line)

# # Example usage:
# modify_annotations(r"labels")

#  modify labels

import os

def modify_annotations(folder_path):
    """
    Modifies the class index in label (.txt) files within a folder.
    Changes class '0' to '80'.

    Args:
        folder_path (str): Path to the folder containing label (.txt) files.
    """
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)

            # Read the lines in the label file
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Modify class '0' to '80' and rewrite the file
            with open(file_path, 'w') as file:
                for line in lines:
                    parts = line.strip().split()
                    if parts[0] == '6':
                        parts[0] = '0'
                    modified_line = ' '.join(parts)
                    file.write(modified_line + '\n')
                    
modify_annotations(r"labels")