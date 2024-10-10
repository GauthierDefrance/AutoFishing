#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      defra
#
# Created:     10/10/2024
# Copyright:   (c) defra 2024
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
from shapely.geometry import Polygon
from PIL import Image, ImageEnhance
from lib.CaptureEcran import *
import time

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Define two convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # Define two fully connected layers (classifier)
        self.fc1 = nn.Linear(32*7*7, 128)
        self.fc2 = nn.Linear(128, 4)  # 4 classes: Z, Q, S, D

    def forward(self, x):
        # Apply ReLU activation to the first conv layer, followed by max pooling
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        # Apply ReLU to the second conv layer, followed by max pooling
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        # Flatten the tensor for the fully connected layer
        x = x.view(-1, 32*7*7)
        x = torch.relu(self.fc1(x))
        # Final output through the second fully connected layer (no activation)
        x = self.fc2(x)
        return x

# ----------------
# Loading the trained model
model = SimpleCNN()  # Instantiate the model
# Load the model weights from 'model.pth'
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Switch to evaluation mode (disable dropout, batchnorm, etc.)
# ----------------

def main():
    window_name = "fishing"  # The starting name of the target window

    # Wait for the user to press 'Enter' to start
    while (detect_key_press() != "enter"):
        print("Waiting to start.")
        time.sleep(0.3)

    # The main loop will run until the user presses 'Esc'
    while (detect_key_press()) != "esc":
        # Capture the images of the circle and the letter
        CircleImage, LetterImage = getImage(window_name)

        # Create an ImageScanner object to process the captured image
        scanner = ImageScanner(CircleImage)

        # Enhance the image (increase brightness, reduce contrast)
        scanner.process_image()

        # Detect collision between the cursor and the blue zone
        if (scanner.detect_collision(scanner.blue_points, scanner.red_points)):
            time.sleep(0.1)  # Short delay before detecting the actual collision
            print("Cursor inside the zone")

            # Use the model to predict the letter (Z, Q, S, or D)
            ToPressTouch = testimg(LetterImage)
            print("Detected Letter: ", ToPressTouch)

            # Simulate pressing the detected key
            presstouch(window_name, ToPressTouch)

    # Program ends when 'Esc' is pressed
    print("Program ended")

# The ImageScanner class is used to process images and detect cursor/zone collisions
class ImageScanner:
    def __init__(self, img):
        # Load and enhance the image (increase brightness and saturation)
        self.image = self.load_and_enhance_image(img)
        # Convert the image to HSV color space for better color detection
        self.hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

    def load_and_enhance_image(self, img):
        """Load and enhance image brightness and saturation."""
        image = img
        # Increase brightness
        image_bright = ImageEnhance.Brightness(image).enhance(0.8)
        # Increase saturation
        image_saturated = ImageEnhance.Color(image_bright).enhance(1.2)
        return cv2.cvtColor(np.array(image_saturated), cv2.COLOR_RGB2BGR)

    def create_mask(self, lower, upper):
        """Create a color mask based on given thresholds."""
        mask = cv2.inRange(self.hsv_image, lower, upper)
        # Erode and dilate to remove noise
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    def find_contours(self, mask):
        """Find and return contours from a binary mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def scan_cursor(self):
        """Detect red areas (representing the cursor) in the image."""
        # Define color ranges for red and magenta
        lower_red_1 = np.array([0, 100, 50])     # Light red
        upper_red_1 = np.array([10, 255, 255])   # Light red
        lower_red_2 = np.array([160, 100, 50])   # Dark red
        upper_red_2 = np.array([180, 255, 255])  # Dark red
        lower_magenta = np.array([140, 100, 50]) # Magenta
        upper_magenta = np.array([160, 255, 255])# Magenta

        # Create binary masks for red and magenta
        mask_red_1 = self.create_mask(lower_red_1, upper_red_1)
        mask_red_2 = self.create_mask(lower_red_2, upper_red_2)
        mask_magenta = self.create_mask(lower_magenta, upper_magenta)

        # Combine the red and magenta masks
        self.mask_red = cv2.bitwise_or(mask_red_1, mask_red_2)
        self.mask_red = cv2.bitwise_or(self.mask_red, mask_magenta)

        # Blur and dilate the mask to reduce noise
        self.mask_red = cv2.GaussianBlur(self.mask_red, (5, 5), 0)
        kernel = np.ones((1, 1), np.uint8)
        self.mask_red = cv2.dilate(self.mask_red, kernel, iterations=2)

        # Find the contours of the red areas (cursor)
        contours_red = self.find_contours(self.mask_red)
        self.red_points = contours_red

    def scan_zone(self):
        """Detect blue areas (representing the click zone) in the image."""
        # Define color ranges for blue
        lower_blue_1 = np.array([80, 150, 0])
        upper_blue_1 = np.array([120, 255, 255])
        lower_blue_2 = np.array([100, 150, 0])
        upper_blue_2 = np.array([140, 255, 255])

        # Create binary masks for blue areas
        mask_blue_1 = self.create_mask(lower_blue_1, upper_blue_1)
        mask_blue_2 = self.create_mask(lower_blue_2, upper_blue_2)
        self.mask_blue = cv2.bitwise_or(mask_blue_1, mask_blue_2)

        # Find the contours of the blue areas (click zones)
        contours_blue = self.find_contours(self.mask_blue)
        self.blue_points = contours_blue

    def get_largest_contour(self, contours):
        """Return the largest contour found."""
        if contours:
            return max(contours, key=cv2.contourArea)
        return []

    def detect_collision(self, contour_a, contour_b):
        """Detect collisions between two sets of contours."""
        for elemA in contour_a:
            ListeA = np.array([point[0] for point in elemA], dtype=np.int32)
            for elemB in contour_b:
                ListeB = np.array([point[0] for point in elemB], dtype=np.int32)

                polygon_a = Polygon(ListeA.reshape(-1, 2)).buffer(1)
                polygon_b = Polygon(ListeB.reshape(-1, 2)).buffer(1)
                # Check if the polygons representing the contours intersect
                if polygon_a.intersects(polygon_b):
                    return True
        return False

    def process_image(self):
        """Process the image to find red and blue areas (cursor and click zone)."""
        self.scan_cursor()
        self.scan_zone()

    def display_masks(self):
        """Display the masks for red and blue areas."""
        if hasattr(self, 'mask_red'):
            cv2.imshow('Mask Red', self.mask_red)
        if hasattr(self, 'mask_blue'):
            cv2.imshow('Mask Blue', self.mask_blue)

    def draw_contours(self):
        """Draw the detected contours on the original image."""
        for contourB, contourR in zip(self.blue_points, self.red_points):
            if cv2.contourArea(contourB) > 10:
                cv2.drawContours(self.image, [contourB], -1, (0, 255, 255), 1)
            if cv2.contourArea(contourR) > 10:
                cv2.drawContours(self.image, [contourR], -1, (255, 255, 0), 1)

    def show_image(self):
        """Display the original image with drawn contours."""
        self.draw_contours()
        cv2.imshow('Original Image', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Function to test an image using the trained model
def testimg(image):
    """Accepts an already opened image and predicts the letter."""
    # Define transformations: convert to grayscale, resize to 28x28, and convert to tensor
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

    # Apply transformations to the image
    image_tensor = transform(image).unsqueeze(0)

    # Use the model to predict the class of the image
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)  # Get the predicted class

    # Map the predicted class index to the corresponding letter
    classes = ['d', 'q', 's', 'z']
    prediction = classes[predicted.item()]

    return prediction  # Return the predicted letter

if __name__ == '__main__':
    main()  # Run the main function
