from skimage.color import rgb2gray
from skimage.segmentation import chan_vese
import matplotlib.pyplot as plt
import cv2 as cv
image=cv.imread("image.jpg")
gray_image=rgb2gray(image)
chanvese_image=chan_vese(gray_image,max_num_iter=100,extended_output=True)
plt.imshow(chanvese_image[1],cmap="gray")
plt.show()