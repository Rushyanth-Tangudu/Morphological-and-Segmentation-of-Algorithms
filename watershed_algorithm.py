import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

# Define the output directory for sure background images
output_directory = r"C:\Users\rushy\OneDrive\Desktop\AMRITA\Sem_5\DIP\dip_review_2\Watershed\sure_background"
os.makedirs(output_directory, exist_ok=True)

# Loop to process 347 images
for i in range(1, 348):  # Adjust the range as necessary
    # Image loading
    img_path = f"C:/Users/rushy/OneDrive/Desktop/AMRITA/Sem_5/DIP/dip_review_1/spatial_filtering/unsharp/unsharp_{i}.jpg"
    img = cv2.imread(img_path)

    if img is None:
        print(f"Error loading image {img_path}")
        continue

    # Image grayscale conversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold Processing
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Noise removal
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background area
    sure_bg = cv2.dilate(bin_img, kernel, iterations=3)

    # Marker labelling
    _, markers = cv2.connectedComponents(bin_img)

    # Add one to all labels so that background is not 0, but 1
    markers += 1
    # Mark the region of unknown with zero
    markers[sure_bg == 255] = 0

    # Save the sure background image
    sure_bg_output_path = os.path.join(output_directory, f"C:/Users/rushy/OneDrive/Desktop/AMRITA/Sem_5/DIP/dip_review_2/watershed/sure_bg_{i}.jpg")
    cv2.imwrite(sure_bg_output_path, sure_bg)

    # Display the results in a subplot format
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    axes[0].imshow(sure_bg, cmap='gray')
    axes[0].set_title('Sure Background')
    axes[0].axis('off')

    axes[1].imshow(markers, cmap="tab20b")
    axes[1].set_title(f'Markers for Image {i}')
    axes[1].axis('off')

    plt.suptitle(f"Processing Image {i}")
    plt.show()

