# Sudoku-Solver
Sudoku puzzle solver 

**Needed installations**
- pip install opencv-python
- pip install numpy
- pip install matplotlib
- pip install urllib

**Phase one done in the following steps**
- Input the image
- Convert to grayscale
- Thresholding step
    - Apply adaptive threshold
- Find contours
    - Filter out smaller contours (potential noise)
    - Find the largest contour (presumably the Sudoku grid)
- Get corners of the largest contour
    - Sort the corners to make sure they are in the correct order for all images
- Perform a perspective transform
- Separate the number tiles