# Sudoku-Solver
Sudoku puzzle solver 

## Needed installations
- pip install opencv-python
- pip install numpy
- pip install matplotlib
- pip install urllib

## Phase one done in the following steps
**1. Input the image**
- read image from a file or url
**2. Convert to grayscale**
- convert the image to grayscale
**3. Thresholding step**
- Apply adaptive threshold
**4. Find contours**
- Filter out smaller contours (potential noise)
- Find the largest contour (presumably the Sudoku grid)
**5. Get corners of the largest contour**
- Sort the corners to make sure they are in the correct order for all images
**6. Perform a perspective transform**
- warp perspective
**7. Separate the number tiles**
- obtain a list of images of the sudoku cells to apply OCR in phase 2

## Output of Phase one
![Phase_1](https://github.com/nadaWagdy/Sudoku-Solver/blob/main/Phase_1.png?raw=true)