import cv2
import numpy as np
import matplotlib.pyplot as mpl

# image file names
# smile_sample_1.tiff
# sample_histogram_equalized.tiff
# /brightness/b_plus_10.tiff
# /brightness/b_min_10.tiff

# Extract histogram for POC
img = cv2.imread('./brightness/b_min_10.tiff', cv2.IMREAD_GRAYSCALE)
histogram = cv2.calcHist([img], [0], None, [256], [0,255])

# displaying
mpl.figure()
mpl.suptitle('Image Histogram')
mpl.tight_layout()
mpl.subplot(1,1,1)
mpl.bar(range(256),histogram.ravel())

# Universal
mpl.show()