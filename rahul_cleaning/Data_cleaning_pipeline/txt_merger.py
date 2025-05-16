import os

def merge_folder_txt_files(folder1, folder2):
    """
    Merges corresponding .txt files from two different folders by appending the content
    of the second folder's file into the first folder's file.

    Args:
        folder1 (str): Path to the first folder.
        folder2 (str): Path to the second folder.
    """
    if not os.path.isdir(folder1) or not os.path.isdir(folder2):
        print(f"One or both directories do not exist: {folder1}, {folder2}")
        return

    # Iterate through the first folder's .txt files
    for file_name in os.listdir(folder1):
        if file_name.endswith('.txt'):
            file1_path = os.path.join(folder1, file_name)
            file2_path = os.path.join(folder2, file_name)

            # If a matching file exists in folder2, append its content to the file in folder1
            if os.path.exists(file2_path):
                with open(file1_path, 'r') as f1:
                    content1 = f1.read()

                with open(file2_path, 'r') as f2:
                    content2 = f2.read()

                with open(file1_path, 'a') as output_file:
                    if not content1.endswith("\n"):
                        output_file.write("\n")
                    output_file.write(content2)

merge_folder_txt_files("labels","annotate")