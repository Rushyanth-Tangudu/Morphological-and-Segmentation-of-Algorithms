import cv2
import os
import matplotlib.pyplot as plt

# Define output base path for edge-detected images
#output_directory_edge = r"C:/Users/rushy/OneDrive/Desktop/AMRITA/Sem_5/DIP/dip_review_1/interpolation/edge_detection"
#os.makedirs(output_directory_edge, exist_ok=True)

# Parameters for Gaussian blur and Canny edge detection
low_threshold = 50
high_threshold = 150
blur_kernel_size = (5, 5)
blur_sigma = 1.4

# Process each image with a defined path in the loop
for i in range(1, 350):  # Loop through images 1 to 349
    # Set the input path for each image
    input_image_path = f"C:/Users/rushy/OneDrive/Desktop/AMRITA/Sem_5/DIP/dip_review_1/spatial_filtering/unsharp/unsharp_{i}.jpg"
    
    # Define output path for each edge-detected image in the specified directory
    #output_image_path = os.path.join(output_directory_edge, f"C:/Users/rushy/OneDrive/Desktop/AMRITA/Sem_5/DIP/dip_review_2/canny_edge_detector/edge_detected_{i}.jpg")

    # Load the image in grayscale
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Error loading image {input_image_path}")
        continue

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(image, blur_kernel_size, blur_sigma)

    # Apply Canny edge detector
    edges = cv2.Canny(blurred_image, low_threshold, high_threshold)

    # Save the edge-detected image in the specified output directory
    #cv2.imwrite(output_image_path, edges)

    # Display the original and edge-detected images
    plt.figure(figsize=(10, 5))

    # Display original image
    plt.subplot(1, 2, 1)
    plt.title(f'Original Image {i}')
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    # Display edge-detected image
    plt.subplot(1, 2, 2)
    plt.title('Canny Edge Detection')
    plt.imshow(edges, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
