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
                    if parts[0] == '0':
                        parts[0] = '80'
                    modified_line = ' '.join(parts)
                    file.write(modified_line + '\n')
                    
modify_annotations(r"annotate")


# change multiple classes
# import os

# def modify_annotations(folder_path, class_mapping):
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".txt"):
#             file_path = os.path.join(folder_path, filename)

#             with open(file_path, 'r') as file:
#                 lines = file.readlines()

#             with open(file_path, 'w') as file:
#                 for line in lines:
#                     parts = line.strip().split()
#                     if parts[0] in class_mapping:
#                         parts[0] = class_mapping[parts[0]]  # Replace the class
#                     modified_line = ' '.join(parts)
#                     file.write(modified_line + '\n')

# # Specify the path to your folder containing the .txt files
# folder_path = r'check'

# # Define a dictionary with the old class as the key and new class as the value
# class_mapping = {
#     '40': '45',
#     '30': '35',
# }

# modify_annotations(folder_path, class_mapping)