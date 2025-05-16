import os
import shutil
from random import choice

# Arrays to store file names
imgs = []
xmls = []

# Setup dir names
trainPath = 'train'
valPath = 'valid'
testPath = 'test'
crsPath = 'data'  # Dir where images and annotations are stored

# Setup ratio (train_ratio + val_ratio + test_ratio = 1.0)
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Sorting files into corresponding arrays
for (dirname, dirs, files) in os.walk(crsPath):
    for filename in files:
        if filename.endswith('.txt'):
            xmls.append(filename)
        elif filename.endswith('.jpg') or filename.endswith('.bmp'):
            imgs.append(filename)

# Counting range for cycles
countForTrain = int(len(imgs) * train_ratio)
countForVal = int(len(imgs) * val_ratio)
countForTest = int(len(imgs) * test_ratio)
print("Training images: ", countForTrain)
print("Validation images: ", countForVal)
print("Test images: ", countForTest)

# Create directories for train, valid, and test sets
trainImagePath = os.path.join(trainPath, 'images')
trainLabelPath = os.path.join(trainPath, 'labels')
valImagePath = os.path.join(valPath, 'images')
valLabelPath = os.path.join(valPath, 'labels')
testImagePath = os.path.join(testPath, 'images')
testLabelPath = os.path.join(testPath, 'labels')

# Create directories if they do not exist
os.makedirs(trainImagePath, exist_ok=True)
os.makedirs(trainLabelPath, exist_ok=True)
os.makedirs(valImagePath, exist_ok=True)
os.makedirs(valLabelPath, exist_ok=True)
os.makedirs(testImagePath, exist_ok=True)
os.makedirs(testLabelPath, exist_ok=True)

# Helper function to check if the file exists
def file_exists(filepath):
    return os.path.isfile(filepath)

# Cycle for train dir
for _ in range(countForTrain):
    fileImg = choice(imgs)  # Get name of random image from origin dir
    fileXml = fileImg.rsplit('.', 1)[0] + '.txt'  # Get name of corresponding annotation file

    # Check if the files exist before copying
    if file_exists(os.path.join(crsPath, fileImg)) and file_exists(os.path.join(crsPath, fileXml)):
        shutil.copy(os.path.join(crsPath, fileImg), os.path.join(trainImagePath, fileImg))
        shutil.copy(os.path.join(crsPath, fileXml), os.path.join(trainLabelPath, fileXml))
        
        # Remove files from arrays
        imgs.remove(fileImg)
        xmls.remove(fileXml)
    else:
        print(f"Warning: Missing file(s) - Image: {fileImg}, XML: {fileXml}")

# Cycle for val dir
for _ in range(countForVal):
    fileImg = choice(imgs)  # Get name of random image from origin dir
    fileXml = fileImg.rsplit('.', 1)[0] + '.txt'  # Get name of corresponding annotation file

    # Check if the files exist before copying
    if file_exists(os.path.join(crsPath, fileImg)) and file_exists(os.path.join(crsPath, fileXml)):
        shutil.copy(os.path.join(crsPath, fileImg), os.path.join(valImagePath, fileImg))
        shutil.copy(os.path.join(crsPath, fileXml), os.path.join(valLabelPath, fileXml))
        
        # Remove files from arrays
        imgs.remove(fileImg)
        xmls.remove(fileXml)
    else:
        print(f"Warning: Missing file(s) - Image: {fileImg}, XML: {fileXml}")

# Cycle for test dir
for _ in range(countForTest):
    fileImg = choice(imgs)  # Get name of random image from origin dir
    fileXml = fileImg.rsplit('.', 1)[0] + '.txt'  # Get name of corresponding annotation file

    # Check if the files exist before copying
    if file_exists(os.path.join(crsPath, fileImg)) and file_exists(os.path.join(crsPath, fileXml)):
        shutil.copy(os.path.join(crsPath, fileImg), os.path.join(testImagePath, fileImg))
        shutil.copy(os.path.join(crsPath, fileXml), os.path.join(testLabelPath, fileXml))
        
        # Remove files from arrays
        imgs.remove(fileImg)
        xmls.remove(fileXml)
    else:
        print(f"Warning: Missing file(s) - Image: {fileImg}, XML: {fileXml}")

print("Splitting into train, valid, and test sets completed successfully.")
