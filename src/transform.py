import numpy as np
import cv2

def orderPoints(points):
    # Defining a list of coordinates (0th is top left, 1st is top right, 2nd is bottom right, 3rd is bottom left)
    rectangle = np.zeros((4,2), dtype="float32")

    sumOfPoints = points.sum(axis = 1)
    rectangle[0] = points[np.argmin(sumOfPoints)] # Top left with smallest sum
    rectangle[2] = points[np.argmax(sumOfPoints)] # Bottom right with largest sum

    difference = np.diff(points, axis = 1)
    rectangle[1] = points[np.argmin(difference)] # Top right with the smallest difference
    rectangle[3] = points[np.argmax(difference)] # Bottom left with the largest difference

    return rectangle

def fourPointTransform(image, points):
    rectangle = orderPoints(points)

    # Sets the values of the coordinates based on the orderPoints function return value of rectangle
    (topLeft, topRight, bottomRight, bottomLeft) = rectangle

    # Calculate the width of the new image which widthA is the maximum width between the bottomRight and bottomLeft and widthB is the maximum width between the topRight and topLeft.
    widthA = np.sqrt(((bottomRight[0] - bottomLeft[0]) ** 2) + ((bottomRight[1] - bottomLeft[1]) ** 2))
    widthB = np.sqrt(((topRight[0] - topLeft[0]) ** 2) + ((topRight[1] - topLeft[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Same as width but with height
    heightA = np.sqrt(((topRight[0] - bottomRight[0]) ** 2) + ((topRight[1] - bottomRight[1]) ** 2))
    heightB = np.sqrt(((topLeft[0] - bottomLeft[0]) ** 2) + ((topLeft[1] - bottomLeft[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Create destination points to make a "birds eye view" with top left, top right, bottom right, bottom left (in order)
    destinationPoints = np.array([[0,0],[maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype = "float32")

    matrix = cv2.getPerspectiveTransform(rectangle, destinationPoints)
    warped = cv2.warpPerspective(image, matrix, (maxWidth, maxHeight))

    return warped