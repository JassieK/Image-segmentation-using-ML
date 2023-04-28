#Preprocessing of fused Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the pansharpened image
image = cv2.imread('D:/fused_img.jpg')

#display
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(image)
ax.axis('off')
plt.show()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply histogram equalization to improve contrast
equalized = cv2.equalizeHist(blur)

# Calculate the mean pixel value of the grayscale image
mean = equalized.mean()

# Calculate the value to add to each pixel to increase the brightness by 50%
delta = int(0.5 * (255 - mean))

# Add the delta value to each pixel using cv2.convertScaleAbs()
brightened = cv2.convertScaleAbs(image, alpha=1, beta=delta) 

# Apply sharpening filter to enhance edges
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpened = cv2.filter2D(brightened, -1, kernel)

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(sharpened)
ax.axis('off')
plt.show()

# Save the preprocessed image
cv2.imwrite('D:/fused_processed.jpg', sharpened)
