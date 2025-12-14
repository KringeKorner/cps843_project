import cv2
import numpy as np

# brightness filter will be based on linear scaling (s = r + b) without a --> not looking to affect contrast directly

img = cv2.imread('./smile_sample_1.tiff', cv2.IMREAD_GRAYSCALE)

def linearScaling (r, b, reduction = False):
    if reduction:
        s = r.astype(np.int16) - b
        s = np.clip(s, 0, 255)
    else:
        s = r.astype(np.int16) + b
        s = np.clip(s, 0, 255)
    return s.astype(np.uint8)

# set to reduce or not
# reduce = False
reduce = True

# set amount to scale
brightness = 50

linear_scaled = linearScaling(img, brightness, reduce).astype(np.uint8)
if reduce:
    cv2.imwrite(f'./brightness/b_min_{brightness}.tiff', linear_scaled)
else:
    cv2.imwrite(f'./brightness/b_plus_{brightness}.tiff', linear_scaled)

while(1):
	cv2.imshow("Altered brightness image", linear_scaled)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
cv2.destroyAllWindows()