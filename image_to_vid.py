import cv2
import os

# Get the path to the images folder
images_folder = "/Users/abhaysangwan/Desktop/G6_gimbal_code/Nvidia_Dataset"

# Get the list of images in the folder
images = os.listdir(images_folder)

# Get the dimensions of the first image
width, height = cv2.imread(os.path.join(images_folder, images[0])).shape[1], cv2.imread(os.path.join(images_folder, images[0])).shape[0]

# Create a video writer object
video_writer = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'DIVX'), 20, (width, height))

# Iterate over the images
for image in images:

    # Read the image
    img = cv2.imread(os.path.join(images_folder, image))

    if img is None:
        print("Error: Image {} could not be loaded.".format(image))
        continue

    # Resize the image to the desired dimensions
    img = cv2.resize(img, (width, height))

    # Write the image to the video writer
    video_writer.write(img)

# Release the video writer
video_writer.release()

# Display a message
print("Video created successfully!")