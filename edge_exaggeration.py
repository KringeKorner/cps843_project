import cv2
import numpy as np
import matplotlib.pyplot as mpl

# contrast will be changed based on the power law transformation for a particular gamma value and employ either log or inverse log based on the gamma
# log and gamma < 1 brightens, inverse log and gamma > 1 darkens

# image file names
# ./smile_sample_1.tiff

file = './smile_sample_1.tiff'
filename = file.split('/')[-1].split('.tiff')[0]

img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
k_sobel = 0.5
k_blur = 10
k_gaussian = 0.5
k_sobel_record = str(k_sobel).replace('.', '_')
k_gaussian_record = str(k_gaussian).replace('.', '_')

# Sobel
def sobelBoost(r, k):
    sobel_gx = cv2.Sobel(r.copy().astype(np.float32), cv2.CV_32F, 1, 0, 3)
    sobel_gy = cv2.Sobel(r.copy().astype(np.float32), cv2.CV_32F, 0, 1, 3)
    sobel_filter = cv2.magnitude(sobel_gx, sobel_gy)
    sobel_mask = sobel_filter / np.max(sobel_filter)
    sobel_mask *= float(k)
    sobel_boosted = r.copy().astype(np.float32) - (k * sobel_mask * r.copy().astype(np.float32))
    return np.clip(sobel_boosted, 0, 255).astype(np.uint8)

# highboost
def highBoost(r, x, y, k):
    img_blur = cv2.blur(r.copy().astype(np.float32), (x, y))
    img_mask = r.copy().astype(np.float32) - img_blur
    img_highboost = r.copy().astype(np.float32) + (k*img_mask)
    return np.clip(img_highboost, 0, 255).astype(np.uint8)

sobel_filter = sobelBoost(img, k_sobel)
highboost_filter = highBoost(img, k_blur, k_blur, k_gaussian)
cv2.imwrite('./edges/' + filename + '_sobel_' + str(k_sobel_record) + '.tiff', sobel_filter)
cv2.imwrite('./edges/' + filename + '_gaussian_' + str(k_gaussian_record) + '.tiff', highboost_filter)

while 1:
    cv2.imshow("Sobel Edge filter", sobel_filter)
    cv2.imshow("Highboost filter", highboost_filter)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()