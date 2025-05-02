from transform import fourPointTransform
import numpy as np
import argparse
import cv2
import imutils
import pytesseract 
import os
from PIL import Image

def main():
    imageInfo = mainScan()
    imageText = imageInfo[0]
    actualImage = imageInfo[1]
    createAndWriteToFile(imageText)
    saveToPDF(actualImage)

def mainScan():
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
    cv2.imshow("Original", image)
    cv2.imshow("Edged", edged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Find countours of the edged image, keep the largest ones, initialize the screen contours
    contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # If the contour has four points then we assume we have found our screen (four corners)
        if len(approx) == 4:
            screenCount = approx
            break

    print("Step 2: Find contours of paper")
    cv2.drawContours(image, [screenCount], -1, (0, 255, 0), 2)
    cv2.imshow("Outline", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Apply the four point transform from transform.py to get a top down view
    warped = fourPointTransform(original, screenCount.reshape(4,2) * ratio)

    # Convert the warped image to grayscale to give it a "black and white" look
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, warped = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    # Show the original and scanned images 
    print("Step 3: Apply perspective transform")
    cv2.imshow("Original", imutils.resize(original, height = 650))
    cv2.imshow("Scanned", imutils.resize(warped, height = 650))
    cv2.waitKey(0)

    warped_for_ocr = imutils.resize(warped, width=1000)
    warped_for_ocr = cv2.cvtColor(warped_for_ocr, cv2.COLOR_GRAY2BGR)

    return pytesseract.image_to_string(warped_for_ocr), imutils.resize(warped, height = 650)

def createAndWriteToFile(infoToWrite):
    fileCount = len(os.listdir("scannedImages/imageTextFiles"))

    fileName = "scannedImages/imageTextFiles/scannedImage" + str((fileCount + 1)) + ".txt"
    imageFile = open(fileName, "w")
    imageFile.write(infoToWrite)

def saveToPDF(imageInfo):
    fileCount = len(os.listdir("scannedImages/imagePDFs"))

    cv2.imwrite("tempImage.png", imageInfo)
    image = Image.open("tempImage.png")

    image_rgb = image.convert("RGB")
    pdfName = f"scannedImages/imagePDFs/scannedImage{fileCount + 1}.pdf"
    image_rgb.save(pdfName, "pdf")

    os.remove("tempImage.png")

main()
