from transform import fourPointTransform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

argumentParse = argparse.ArgumentParser()
argumentParse.add_argument("-i", "--image", required = True, help = "Path to the image to be scanned")
args = vars(argumentParse.parse_args())

# Load image, compute ratio of old height, copy it, resize it
image = cv2.imread(args["image"])
ratio = image.shape[0] / 500.0
original = image.copy()
image = imutils.resize(image, height = 500)

# Convert to grayscale, blur it, find edges
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)
edged = cv2.Canny(gray, 75, 200)

print("Step 1: Edge Detection")
cv2.imshow("Edged.png", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()
