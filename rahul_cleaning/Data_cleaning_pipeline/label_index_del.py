# import os

# def modify_annotations(folder_path):
#     """
#     Modifies the label (.txt) files within a folder by removing lines with class '80'.

#     Args:
#         folder_path (str): Path to the folder containing label (.txt) files.
#     """
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".txt"):
#             file_path = os.path.join(folder_path, filename)

#             # Read the lines in the label file
#             with open(file_path, 'r') as file:
#                 lines = file.readlines()

#             # Rewrite the file without lines containing class '80'
#             with open(file_path, 'w') as file:
#                 for line in lines:
#                     parts = line.strip().split()
#                     # Only write lines that do not start with '80'
#                     if parts[0] != '80':
#                         file.write(line)  # Write the original line back

# # Example usage:
# modify_annotations(r"labels")

################################################################################################

import os

def modify_annotations(folder_path, classes_to_remove):
    """
    Modifies the label (.txt) files within a folder by removing lines with specified classes.

    Args:
        folder_path (str): Path to the folder containing label (.txt) files.
        classes_to_remove (list): List of class IDs (as strings) to remove from the annotations.
    """
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)

            # Read the lines in the label file
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Rewrite the file without lines containing specified classes
            with open(file_path, 'w') as file:
                for line in lines:
                    parts = line.strip().split()
                    # Only write lines that do not start with any of the specified classes
                    if parts[0] not in classes_to_remove:
                        file.write(line)  # Write the original line back

# Example usage:
modify_annotations(r"labels", classes_to_remove=['4','7', '8','9','10','11','12','13','22','22', '23', '24', '25', '26', '27', '28', '29', '30','31', '32', '33', '34', '35', '36', '37', '38', '39', '40','41', '42', '43', '44', '45', '46', '47', '48', '49', '50','51', '52','53', '54', '55', '56', '57', '58', '59', '60','61', '62', '63', '64', '65', '66', '67', '68', '69', '70','71', '72', '73', '74', '75', '76', '77', '78', '79'])  # Add more classes as needed
