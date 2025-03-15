import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import cv2 as cv
from skimage.filters import gaussian
from skimage.segmentation import active_contour
image=cv.imread("image.jpg")
gray_image = rgb2gray(image)
gray_image_noiseless = gaussian(gray_image, 1)
x1 = 220 + 100 * np.cos(np.linspace(0, 2 * np.pi, 500))
x2 = 100 + 100 * np.sin(np.linspace(0, 2 * np.pi, 500))
snake = np.array([x1, x2]).T
image_snake = active_contour(gray_image_noiseless, snake)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
ax.imshow(gray_image_noiseless)
ax.plot(image_snake[:, 0], image_snake[:, 1], '-b', lw=5)
ax.plot(snake[:, 0], snake[:, 1], '--r', lw=5)
plt.show()