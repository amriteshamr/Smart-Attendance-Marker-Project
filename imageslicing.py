# This program recieves an image and slices it into 4*4 images and saves it separately
 
import cv2

# Load the image
image = cv2.imread("image.jpg")

# Get the dimensions of the image
height, width, _ = image.shape

# Calculate the dimensions of each piece
piece_height = height // 4
piece_width = width // 4

# Iterate through each piece
for row in range(4):
    for col in range(4):
        # Get the coordinates of the piece
        x1 = col * piece_width
        y1 = row * piece_height
        x2 = x1 + piece_width
        y2 = y1 + piece_height

        # Crop the image to get the piece
        piece = image[y1:y2, x1:x2]

        # Save the piece as a separate image
        cv2.imwrite("piece_{}_{}.jpg".format(row, col), piece)
