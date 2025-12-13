import cv2
import numpy as np
import matplotlib.pyplot as mpl

# contrast will be changed based on the power law transformation for a particular gamma value and employ either log or inverse log based on the gamma
# log and gamma < 1 brightens, inverse log and gamma > 1 darkens

# image file names
# ./smile_sample_1.tiff
# ./sample_histogram_equalized.tiff
# ./brightness/b_plus_10.tiff
# ./brightness/b_min_10.tiff

file = './smile_sample_1.tiff'
filename = file.split('/')[-1].split('.tiff')[0]

img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

def logTransform(r):
    r = r.astype(np.float32)
    c = 255 / np.log(1 + 255)
    s = c * np.log(1 + r)
    return np.clip(s, 0, 255).astype(np.uint8)

def logInverseTransform(r):
    r = r.astype(np.float32) / 255.0
    s = np.exp(r) - 1
    s = s / np.max(s)
    s = s * 255
    return np.clip(s, 0, 255).astype(np.uint8)

def powerLaw(r, g):
    r = r.astype(np.float32) / 255.0
    s = np.power(r, gamma)
    s = s * 255
    return np.clip(s, 0, 255).astype(np.uint8)

gamma = 0.5

if gamma < 1:
    log = logTransform(img)
    power_law = powerLaw(img, gamma)
    cv2.imwrite('./log/' + filename + '_log.tiff', log)
    cv2.imwrite('./powerlaw/' + filename + '_gamma_' + str(gamma) + '.tiff', power_law)
else: 
    inverse = logInverseTransform(img)
    power_law = powerLaw(img, gamma)
    cv2.imwrite('./inverselog/' + filename + '_inv.tiff', inverse)
    cv2.imwrite('./powerlaw/' + filename + '_gamma_' + str(gamma) + '.tiff', power_law)

while 1:
    if gamma < 1:
        cv2.imshow("Log Contrast filter", log)
    else:
        cv2.imshow("Inverse Log Contrast filter", inverse)

    cv2.imshow("Power Law Contrast filter", power_law)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()