import os

folder = "background"
count = 1

for root, dirs, files in os.walk(folder):
    for filename in files:
        ext = os.path.splitext(filename)[1]
        new_name = f"Background{count:03d}{ext}"
        old_path = os.path.join(root, filename)
        new_path = os.path.join(root, new_name)
        os.rename(old_path, new_path)
        print(f"{old_path} → {new_path}")
        count += 1


# import os
# from pathlib import Path

# def rename_images_recursive(main_folder):
#     main_path = Path(main_folder)
#     global_counter = 1

#     for img_file in sorted(main_path.rglob("*")):
#         if img_file.is_file() and img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"]:
#             # Build a folder-based name for context (flatten path)
#             relative_folder = img_file.parent.relative_to(main_path)
#             folder_name = "_".join(relative_folder.parts)
            
#             ext = img_file.suffix.lower()
#             new_name = f"{global_counter:05d}{ext}"
#             new_path = img_file.parent / new_name

#             try:
#                 img_file.rename(new_path)
#                 print(f"Renamed: {img_file} → {new_name}")
#                 global_counter += 1
#             except Exception as e:
#                 print(f"Error renaming {img_file}: {e}")

# if __name__ == "__main__":
#     rename_images_recursive("wintagge")  # Change to your image folder path

