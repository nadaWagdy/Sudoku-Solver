import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request

M = 9

# Input: sudoku grid ( 9 * 9 list )
# function to print sudoku grid
def print_sudoku_grid(sudoku_grid):
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("-"*21)
        for j in range(9):
            if j % 3 == 0 and j != 0:
                print("|", end=" ")
            print(sudoku_grid[i][j], end=" ")
        print()


# Input: sudoku grid - starting row - starting column
# Output: returns boolean if sudoku is solvable
# function to solve sudoku
def solve(grid, row, col, num):
    for x in range(9):
        if grid[row][x] == num:
            return False

    for x in range(9):
        if grid[x][col] == num:
            return False

    startRow = row - row % 3
    startCol = col - col % 3
    for i in range(3):
        for j in range(3):
            if grid[i + startRow][j + startCol] == num:
                return False
    return True


# Input: sudoku grid - row - column
# Output: returns boolean if sudoku is solvable
# function to solve sudoku
def Suduko(grid, row, col):
    if (row == M - 1 and col == M):
        return True
    if col == M:
        row += 1
        col = 0
    if grid[row][col] > 0:
        return Suduko(grid, row, col + 1)
    for num in range(1, M + 1, 1):

        if solve(grid, row, col, num):

            grid[row][col] = num
            if Suduko(grid, row, col + 1):
                return True
        grid[row][col] = 0
    return False


# Output: templates of digits to match
# function to create digit templates in order to match them with grid
def create_digit_templates():
    digit_templates = []
    for i in range(1, 10):
        template = np.zeros((40, 40), dtype=np.uint8)
        if i != 3:
            cv2.putText(template, str(i), (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, 255, 2)
        else:
            cv2.putText(template, str(i), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, 255, 2)
        digit_templates.append(template)

    return digit_templates


# Input: boxes (a list of cells image) - digit_templates (templates of digits to match) - min_confidence (minimum confidence in matching)
# Output: a 9 * 9 list of exctracted numbers from input images
# function to match cells images with templates
def extract_numbers(boxes, digit_templates, min_confidence=0.7):
    extracted_numbers = []

    for box_row in range(9):
        row_numbers = []
        for box_col in range(9):
            box = boxes[box_row * 9 + box_col]
            if np.sum(box > 200) > box.size * 0.1:
                detected_digit = None
                max_confidence = -1

                # Ensure that box and template have the same data type (np.uint8)
                box = box.astype(np.uint8)


                for i, template in enumerate(digit_templates):
                    template = template.astype(np.uint8)  # Ensure template data type
                    result = cv2.matchTemplate(box, template, cv2.TM_CCOEFF_NORMED)
                    _, confidence, _, _ = cv2.minMaxLoc(result)

                    # Normalize confidence to the range [0, 1]
                    confidence = (confidence + 1) / 2

                    if confidence > max_confidence:
                        max_confidence = confidence
                        detected_digit = i + 1

                if max_confidence >= min_confidence:
                    row_numbers.append(detected_digit)
                else:
                    row_numbers.append(0)  # Use 0 to represent empty cells
            else:
                row_numbers.append(0)

        extracted_numbers.append(row_numbers)

    return extracted_numbers


# Input: corners of grid
# Output: list of ordered corners
# function to sort the detected corners of the grid in the correct order
def reorder_corners(points):
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]
    new_points[2] = points[np.argmax(add)]  # 2
    diff = np.diff(points, axis=1)
    new_points[3] = points[np.argmin(diff)]  # 3
    new_points[1] = points[np.argmax(diff)]  # 1
    return new_points

# Step 1: Input the image
# in case of url - uncomment the following lines
# req = urllib.request.urlopen('https://live.staticflickr.com/8027/6978422072_33ac92fe1a_b.jpg')
# arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
# original_image = cv2.imdecode(arr, -1)  # 'Load it as it is'

original_image = cv2.imread('test cases/08-MeshShayef7agaYa3am.jpg')

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
            # Sort the corners to make sure they are in the correct order for all images
            corners = reorder_corners(corners)
        else:
            print("Invalid shape of corners. Expected (4, 1, 2)")
    else:
        print("Insufficient number of points in corners")

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
    boxes = []

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
            boxes.append(cell_content)

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

    # create the digit templates and match them with sudoku cells
    digit_templates = create_digit_templates()
    numbers = extract_numbers(boxes, digit_templates)
    print("")
    print("Detected Soduku Grid")
    print_sudoku_grid(numbers)

    print("")
    print("**************************************************************")
    print("")

    # Solve Sudoku
    if (Suduko(numbers, 0, 0)):
        print("Solved Sudoku Puzzle")
        print_sudoku_grid(numbers)
    else:
        print("Sudoku grid not solvable. Check input again.")
else:
    print("No Sudoku grid detected.")