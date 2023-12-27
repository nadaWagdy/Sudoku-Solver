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

## Phase two done in the following steps
**1. Create digits templates**
- use numpy to create digit templates

**2. Match extracted cells from phase one with templates**
- compute confidence in matching and exctract accordingly

**3. Extract the numbers**
- extract numbers in a 9 * 9 list based on the confidence computed

**4. Check if sudoku is solvable**

**5. Solve Sudoku Grid**
- solve sudoku puzzle if it is proved to be solvable then output the solved grid

## Sample Output for phase two
![Phase_2](https://github.com/nadaWagdy/Sudoku-Solver/blob/main/Phase_2.png?raw=true)

**Detected Sudoku Grid**

![2_Detected_Grid](https://github.com/nadaWagdy/Sudoku-Solver/blob/main/2_Detected_Grid.png?raw=true)


**Solved Sudoku Grid**

![2_Solved_Grid](https://github.com/nadaWagdy/Sudoku-Solver/blob/main/2_Solved_Grid.png?raw=true)