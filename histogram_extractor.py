import cv2
import numpy as np
import matplotlib.pyplot as mpl

# image file names
# ./smile_sample_1.tiff
# ./equalized/smile_sample_1_equalized.tiff
# ./brightness/b_plus_10.tiff
# ./brightness/b_min_10.tiff
# ./log/smile_sample_1_log.tiff
# ./inverselog/smile_sample_1_inv.tiff
# ./powerlaw/smile_sample_1_gamma_0_5.tiff
# ./powerlaw/smile_sample_1_gamma_1_2.tiff
# ./powerlaw/smile_sample_1_gamma_1_8.tiff

# Extract histogram for POC
img = cv2.imread('./smile_sample_1_equalized.tiff', cv2.IMREAD_GRAYSCALE)
histogram = cv2.calcHist([img], [0], None, [256], [0,255])

# displaying
mpl.figure()
mpl.suptitle('Image Histogram')
mpl.tight_layout()
mpl.subplot(1,1,1)
mpl.bar(range(256),histogram.ravel())

# Universal
mpl.show()