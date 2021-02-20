import cv2 as cv
import numpy as np

img = np.array([[0, 1, 2, 3],
               [4, 5, 6, 7],
               [8, 9, 0, 1],
               [2, 3, 4, 5]], dtype=np.uint8)

kernel = np.array([[0,1,0],
                   [1,2,1],
                   [0,1,0]])

print(cv.filter2D(img, -1, kernel))