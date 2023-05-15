import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2

#ualize and print each cluster
#for i in range(k):
image1 = plt.imread('D:/IMAGEE SEG/Registered imgs/2j/2j_fused.jpg')  # Load the satellite image
image1gray = cv2.cvtColor(image1,code=cv2.COLOR_RGB2GRAY)  # Load the satellite image grayscale
image2 = plt.imread('D:/IMAGEE SEG/Registered imgs/2j/2j_enhanced.jpg')  # Load the enhanced satellite image
sr = np.ones((5,5))
Idil = cv2.dilate(image1gray,sr)
Ier = cv2.erode(image1gray,sr)
I2 = Idil-Ier
# Normalize the pixel values to range between 0 and 1
image = np.stack([image1[:,:,0],image1[:,:,1],image1[:,:,2],image2[:,:,0],image2[:,:,1],image2[:,:,2],I2],axis=2) / 255.0

# Reshape the image to a 2D array
pixels = image.reshape(-1, 7)

k = 6  # Number of clusters
kmeans = KMeans(n_clusters=k)
kmeans.fit(pixels)
colors = [[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255],[255,0,255]]
labels = kmeans.labels_
labels = labels.reshape(image.shape[0], image.shape[1])
labels = np.stack([labels,labels,labels],axis=2)
for i in range(len(labels)):
    for j in range(len(labels[0])):
        labels[i][j] = colors[labels[i][j][0]]
plt.imsave('D:/IMAGEE SEG/Registered imgs/2j/2j_segmented.jpg',labels.astype(np.uint8))
#fig, ax = plt.subplots(figsize=(20, 20))
#ax.imshow(labels)
#ax.axis('off')
#plt.show()

# Vis
fig, ax = plt.subplots(figsize=(10, 10))
cluster = np.zeros_like(image)
cluster[labels == i] = image[labels == i]
plt.imshow(cluster)
#plt.imsave('D:/Cluster'+str(i)+'.jpg',cluster)
plt.title('Cluster {}'.format(i))
plt.show()