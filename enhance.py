import cv2
import numpy as np
from scipy.signal import correlate2d as corr, convolve2d as conv
from matplotlib import pyplot as plt



I = cv2.imread('D:/IMAGEE SEG/Registered imgs/2j/2j_segmented.jpg')
kernel = np.array([[0, -1, 0],
                [-1, 5,-1],
                [0, -1, 0]])
kernel2 = np.array([[1, 2, 4, 2, 1],
                    [2, 4, 8, 4, 2],
                    [4, 8, 16, 8, 4],
                    [2, 4, 8, 4, 2],
                    [1, 2, 4, 2, 1]])/100
#Isharp = cv2.filter2D(src=I, ddepth=-1, kernel=kernel)
#sr = np.ones((5,5))
#Idil = cv2.dilate(I,sr)
#Ier = cv2.erode(I,sr)
#I2 = Idil-Ier
#Isharp = Isharp | I2
#Isharp = Isharp ^ I2
Iblur = cv2.filter2D(src=I, ddepth=-1, kernel=kernel2)
cv2.imwrite('D:/IMAGEE SEG/Registered imgs/2j/2j_seg_blurred.jpg',Iblur)
