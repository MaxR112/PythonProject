from transform import fourPointTransform
import numpy as np
import argparse
import cv2

argumentParse = argparse.ArgumentParser()
argumentParse.add_argument("-i", "--image", help = "path to the image file")
argumentParse.add_argument("-c", "--coords", help = "comma seperated list of source points")
args = vars(argumentParse.parse_args())

image = cv2.imread(args["image"])
points = np.array(eval(args["coords"]), dtype = "float32")

warped = fourPointTransform(image, points)

cv2.imwrite("Original.png", image)
cv2.imwrite("Warped.png", warped)
cv2.waitKey(0)