import cv2
import matplotlib.pyplot as plt  
import numpy as np

#Reading The images
multi= cv2.imread("D:\IMAGEE SEG\Processed imgs\Multispectral.jpg")
multi = cv2.cvtColor(multi , cv2.COLOR_BGR2RGB)
panc = cv2.imread("D:\IMAGEE SEG\Processed imgs\Panchromatic.jpg")

#plotting the images
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(multi)
ax.axis('off')
plt.show()

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(panc)
ax.axis('off')
plt.show()

# Initialize the feature detector and matcher
detector = cv2.ORB_create()
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Detect features and compute descriptors for both images
kp1, des1 = detector.detectAndCompute(panc, None)
kp2, des2 = detector.detectAndCompute(multi, None)

# Match the features between the two images
matches = matcher.match(des1, des2)

# Compute the homography transformation using RANSAC
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Define the bounding box coordinates of the region of interest in the first image
x, y, w, h = 0, 0, 1700, 1700
roi_pts = np.float32([[x, y], [x+w, y], [x+w, y+h], [x, y+h]]).reshape(-1, 1, 2)

# Apply the homography transformation to the coordinates of the bounding box in the first image
dst_roi_pts = cv2.perspectiveTransform(roi_pts, M)

# Determine the bounding box coordinates of the region of interest in the second image
x_min, y_min = np.min(dst_roi_pts, axis=0).ravel()
x_max, y_max = np.max(dst_roi_pts, axis=0).ravel()
w = x_max - x_min
h = y_max - y_min

# Crop the region of interest from the second image
crop_img = multi[int(y_min):int(y_min+h), int(x_min):int(x_min+w)]

#display
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(crop_img)
ax.axis('off')
plt.show()

# Rescale the panchromatic image to match the range of the multispectral image
pan_rescaled = cv2.resize(panc, (crop_img.shape[1], crop_img.shape[0]), interpolation=cv2.INTER_LINEAR)

# Compute the Brovey transform
weights = pan_rescaled / np.sum(pan_rescaled)
ms_weighted = crop_img.astype(np.float32) * weights.astype(np.float32)
fused_img = np.sum(ms_weighted, axis=2)

# Convert the fused image to uint8 format
fused_img = (fused_img / np.max(fused_img)) * 255
fused_img = fused_img.astype(np.uint8)

#Showing Fused Image
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(fused_img)
ax.axis('off')
plt.show()

cv2.imwrite('fused_img.jpg',fused_img)
