from modules.file_mover import move_files_based_on_classes
from modules.label_modifier import modify_annotations
from modules.image_label_checker import check_image_label_match
from modules.unwanted_label_remover import remove_unwanted_labels
from modules.txt_merger import merge_folder_txt_files

def run_pipeline():
    """
    Runs the full pipeline for moving files, modifying labels, checking image-label matches,
    removing unwanted labels, and merging .txt files.
    """
    # Move files based on target classes
    dataset_dir = "labels"
    image_dir = "test2017"
    destination_dir = "vehicle"
    target_classes = {'1', '2', '3', '5', '7'}
    class_counts, blank_file_count = move_files_based_on_classes(dataset_dir, image_dir, destination_dir, target_classes)
    print("Class counts:", class_counts)
    print("Blank files deleted:", blank_file_count)

    # Modify annotations to change class '0' to '80'
    modify_annotations(destination_dir)

    # Check for unmatched images and move them
    source_image_folder_path = "images"
    label_folder_path = "labels"
    destination_folder_path = "data_vehicle_unmatch"
    check_image_label_match(source_image_folder_path, label_folder_path, destination_folder_path)

    # Remove unwanted label files with no corresponding images
    image_folder_path = "data"
    remove_unwanted_labels(image_folder_path, destination_dir)

    # Merge .txt files from two folders
    folder1 = "vehicle"
    folder2 = "plate"
    merge_folder_txt_files(folder1, folder2)

if __name__ == "__main__":
    run_pipeline()
