import cv2

# image file names
# ./smile_sample_1.tiff
# ./brightness/b_plus_10.tiff
# ./brightness/b_min_10.tiff

# Extract histogram for POC
file = './smile_sample_1.tiff'
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