import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request

# Input: corners of grid
# Output: list of ordered corners
# function to sort the detected corners of the grid in the correct order
def reorder_corners(points):
    print("Original points size:", points.shape)
    points = points.reshape((4, 2))
    print("After points size:", points.shape)
    new_points = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]
    new_points[2] = points[np.argmax(add)]  # 2
    diff = np.diff(points, axis=1)
    new_points[3] = points[np.argmin(diff)]  # 3
    new_points[1] = points[np.argmax(diff)]  # 1
    return new_points

# test cases to work on: 3, 5, 8, 15, 16

# Step 1: Input the image
# in case of url
# req = urllib.request.urlopen('https://live.staticflickr.com/8027/6978422072_33ac92fe1a_b.jpg')
# arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
# original_image = cv2.imdecode(arr, -1)  # 'Load it as it is'

original_image = cv2.imread('test cases/03-WhereBorder.jpg')

# Step 2: Convert to grayscale
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

if np.mean(gray_image) < 100:
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_image = clahe.apply(gray_image)\

# Apply gaussian blur to get rid of the noise
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Step 3: Thresholding step
# Apply adaptive threshold
binary_image = cv2.adaptiveThreshold(
    blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2
)

# Step 4: Find contours
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter out smaller contours (potential noise)
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]

# Find the largest contour (presumably the Sudoku grid)
if filtered_contours:
    largest_contour = max(filtered_contours, key=cv2.contourArea)

    # Step 5: Get corners of the largest contour
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    corners = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(corners) >= 4:
        # If corners has more than 4 points, take the first 4
        corners = corners[:4]

        # Ensure corners has the correct shape (4, 1, 2)
        if corners.shape == (4, 1, 2):
            corners = reorder_corners(corners)
        else:
            print("Invalid shape of corners. Expected (4, 1, 2)")
    else:
        print("Insufficient number of points in corners")
    # Sort the corners to make sure they are in the correct order for all images
    # corners = reorder_corners(corners)

    # Step 6: Perform a perspective transform on the binary image
    pts1 = np.float32(corners)
    pts2 = np.float32([[0, 0], [0, 450], [450, 450], [450, 0]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result_image = cv2.warpPerspective(binary_image, matrix, (450, 450))

    # Step 7: Separate the number tiles
    cell_size = result_image.shape[0] // 9
    cell_border_size = 2

    # Create an array to store the combined cells image
    combined_cells_image = np.ones((450 + 8 * cell_border_size, 450 + 8 * cell_border_size), dtype=np.uint8) * 255

    for row in range(9):
        for col in range(9):
            cell_content = result_image[row * cell_size:(row + 1) * cell_size,
                           col * cell_size:(col + 1) * cell_size].copy()
            cell_content[-cell_border_size:, :] = 255
            cell_content[:, -cell_border_size:] = 255

            y_offset = row * (cell_size + cell_border_size)
            x_offset = col * (cell_size + cell_border_size)

            combined_cells_image[y_offset:y_offset + cell_size,
            x_offset:x_offset + cell_size] = cell_content

    plt.figure(figsize=(15, 10))

    plt.subplot(3, 4, 1), plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    plt.subplot(3, 4, 2), plt.imshow(binary_image, cmap='gray'), plt.title('Binary Image')
    plt.subplot(3, 4, 3), plt.imshow(cv2.drawContours(original_image.copy(), contours, -1, (0, 255, 0), 2)), plt.title('Detected Contours')
    plt.subplot(3, 4, 4), plt.imshow(cv2.drawContours(original_image.copy(), [largest_contour], -1, (0, 255, 0), 2)), plt.title('Sudoku Grid')
    plt.subplot(3, 4, 5), plt.imshow(cv2.drawContours(original_image.copy(), [corners], -1, (0, 255, 0), 2)), plt.title('Detected Corners')
    plt.subplot(3, 4, 6), plt.imshow(result_image, cmap='gray'), plt.title('Perspective Transform')

    plt.subplot(3, 4, 7), plt.imshow(combined_cells_image, cmap='gray'), plt.title('Combined Cells')
    plt.axis('off')

    plt.show()
else:
    print("No Sudoku grid detected.")
