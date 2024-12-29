import cv2
import numpy as np
import matplotlib.pyplot as plt

# Process each image
for i in range(1, 348):  # Loop through images 1 to 347
    # Define the input image path
    image_path = f"C:/Users/rushy/OneDrive/Desktop/AMRITA/Sem_5/DIP/dip_review_1/spatial_filtering/unsharp/unsharp_{i}.jpg"
    
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Error loading image {image_path}")
        continue

    # Apply Gaussian blur to smooth the image
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0.6)

    # Apply the Laplacian operator
    laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)

    # Convert the result to uint8 type for display purposes
    laplacian = cv2.convertScaleAbs(laplacian)

    # Save the LoG filtered image to the current working directory
    output_image_path = f"C:/Users/rushy/OneDrive/Desktop/AMRITA/Sem_5/DIP/dip_review_2/LoG/LoG_filtered_{i}.jpg"
    cv2.imwrite(output_image_path, laplacian)

    # Display the original and LoG-filtered images
    plt.figure(figsize=(10, 10))

    # Display the original image
    plt.subplot(1, 2, 1)
    plt.title(f"Original Image {i}")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    # Display the LoG filtered image
    plt.subplot(1, 2, 2)
    plt.title("Laplacian of Gaussian (LoG) Edge Detection")
    plt.imshow(laplacian, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
