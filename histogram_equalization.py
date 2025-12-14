import cv2

# image file names
# ./smile_sample_1.tiff
# ./brightness/b_plus_10.tiff
# ./brightness/b_min_10.tiff
# ./brightness/b_min_50.tiff
# ./edges/smile_sample_1_sobel_0_5.tiff
# ./edges/smile_sample_1_sobel_1_2.tiff
# ./edges/smile_sample_1_sobel_1_5.tiff
# ./edges/smile_sample_1_sobel_2.tiff
# ./edges/smile_sample_1_gaussian_0_5.tiff
# ./edges/smile_sample_1_gaussian_1_5.tiff
# ./edges/smile_sample_1_gaussian_3.tiff
# ./edges/smile_sample_1_gaussian_5.tiff

# Extract histogram for POC
file = './brightness/b_min_50.tiff'
filename = file.split('/')[-1].split('.tiff')[0]

img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
equalized = cv2.equalizeHist(img)
cv2.imwrite('./equalized/' + filename + '_equalized.tiff', equalized)

while(1):
	cv2.imshow("Equalized Histogram filter", equalized)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
cv2.destroyAllWindows()