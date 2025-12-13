import argparse
import cv2
import os
import cv2

# Haar cascades URL: https://github.com/opencv/opencv/tree/3.4/data/haarcascades

root_path = os.path.dirname(os.path.abspath(__file__))

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
# ./edges/smile_sample_1_sobel_0_5.tiff
# ./edges/smile_sample_1_sobel_1_2.tiff
# ./edges/smile_sample_1_sobel_1_5.tiff
# ./edges/smile_sample_1_sobel_2.tiff
# ./edges/smile_sample_1_gaussian_0_5.tiff
# ./edges/smile_sample_1_gaussian_1_5.tiff
# ./edges/smile_sample_1_gaussian_3.tiff
# ./edges/smile_sample_1_gaussian_5.tiff

# conversion
frame = cv2.imread('./edges/smile_sample_1_sobel_0_5.tiff', cv2.IMREAD_COLOR)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascades", type=str, default="cascades",
	help="path to input directory containing haar cascades")
args = vars(ap.parse_args())

detectorPaths = {
	"face": "haarcascade_frontalface_default.xml",
	"eyes": "haarcascade_eye.xml",
	"smile": "haarcascade_smile.xml",
}

print("[INFO] loading haar cascades...")
detectors = {}

for (name, path) in detectorPaths.items():
	
	path = os.path.join(root_path, args["cascades"], path)
	detectors[name] = cv2.CascadeClassifier(path)

faceRects = detectors["face"].detectMultiScale(
	gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30),
	flags=cv2.CASCADE_SCALE_IMAGE)
for (fX, fY, fW, fH) in faceRects:
	faceROI = gray[fY:fY+ fH, fX:fX + fW]
	eyeRects = detectors["eyes"].detectMultiScale(
		faceROI, scaleFactor=1.1, minNeighbors=10,
		minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE)
	smileRects = detectors["smile"].detectMultiScale(
		faceROI, scaleFactor=1.1, minNeighbors=10,
		minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE)
	for (eX, eY, eW, eH) in eyeRects:
		ptA = (fX + eX, fY + eY)
		ptB = (fX + eX + eW, fY + eY + eH)
		cv2.rectangle(frame, ptA, ptB, (0, 0, 255), 2)
	for (sX, sY, sW, sH) in smileRects:
		ptA = (fX + sX, fY + sY)
		ptB = (fX + sX + sW, fY + sY + sH)
		cv2.rectangle(frame, ptA, ptB, (255, 0, 0), 2)
	cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH),
		(0, 255, 0), 2)
	
while(1):
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
cv2.destroyAllWindows()