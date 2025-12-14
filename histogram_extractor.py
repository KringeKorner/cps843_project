import cv2
import numpy as np
import matplotlib.pyplot as mpl

# image file names
# ./smile_sample_1.tiff
# ./equalized/smile_sample_1_equalized.tiff
# ./equalized/b_min_50_equalized.tiff
# ./equalized/smile_sample_1_sobel_1_2_equalized.tiff
# ./equalized/smile_sample_1_gaussian_1_5_equalized.tiff
# ./equalized/smile_sample_1_gaussian_3_equalized.tiff
# ./equalized/smile_sample_1_gaussian_5_equalized.tiff
# ./brightness/b_plus_10.tiff
# ./brightness/b_min_10.tiff
# ./brightness/b_min_50.tiff
# ./log/smile_sample_1_log.tiff
# ./inverselog/smile_sample_1_inv.tiff
# ./powerlaw/smile_sample_1_gamma_0_5.tiff
# ./powerlaw/smile_sample_1_gamma_1_2.tiff
# ./powerlaw/smile_sample_1_gamma_1_8.tiff
# ./powerlaw/smile_sample_1_gamma_2_5.tiff
# ./edges/smile_sample_1_sobel_0_5.tiff
# ./edges/smile_sample_1_sobel_1_2.tiff
# ./edges/smile_sample_1_sobel_1_5.tiff
# ./edges/smile_sample_1_sobel_2.tiff
# ./edges/smile_sample_1_gaussian_0_5.tiff
# ./edges/smile_sample_1_gaussian_1_5.tiff
# ./edges/smile_sample_1_gaussian_3.tiff
# ./edges/smile_sample_1_gaussian_5.tiff

# Extract histogram for POC
file = './edges/smile_sample_1_gaussian_5.tiff'
filename = file.split('/')[-1].split('.tiff')[0]
img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
histogram = cv2.calcHist([img], [0], None, [256], [0,255])

# displaying
mpl.figure()
mpl.suptitle('Image Histogram')
mpl.tight_layout()
mpl.subplot(1,1,1)
mpl.bar(range(256),histogram.ravel())
# save
mpl.savefig('./histogram/' + filename + '_hist.png')
# Universal
mpl.show()