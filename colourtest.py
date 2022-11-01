# Importing all modules
import cv2
import numpy as np


cv2.namedWindow('HSV')

def empty(a):
    pass


cv2.resizeWindow('HSV', 720, 400)
cv2.createTrackbar('Hue Min', 'HSV', 0, 179, empty)
cv2.createTrackbar('Hue Max', 'HSV', 179, 179, empty)
cv2.createTrackbar('SAT Min', 'HSV', 0, 255, empty)
cv2.createTrackbar('SAT Max', 'HSV', 0, 255, empty)
cv2.createTrackbar('VALUE Min', 'HSV', 0, 255, empty)
cv2.createTrackbar('VALUE Max', 'HSV', 0, 255, empty)


# Capturing webcam footage
# webcam_video = cv2.VideoCapture(0)

while True:
   # success, video = webcam_video.read()  # Reading webcam footage
    img_color = cv2.imread('test.jpg', cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)  # Converting BGR image to HSV format

    h_min = cv2.getTrackbarPos('Hue Min', 'HSV')
    h_max = cv2.getTrackbarPos('Hue Max', 'HSV')
    s_min = cv2.getTrackbarPos('SAT Min', 'HSV')
    s_max = cv2.getTrackbarPos('SAT Max', 'HSV')
    v_min = cv2.getTrackbarPos('VALUE Min', 'HSV')
    v_max = cv2.getTrackbarPos('VALUE Max', 'HSV')

# Specifying upper and lower ranges of color to detect in hsv format
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])  # (These ranges will detect

    mask = cv2.inRange(img, lower, upper)  # Masking the image to find our color
    # Finding contours in mask image
    mask_contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = cv2.bitwise_and(img_color, img_color, mask = mask)
    hstack = np.hstack([img, result])
    # Finding position of all contours
    if len(mask_contours) != 0:
        for mask_contour in mask_contours:
            if cv2.contourArea(mask_contour) > 500:
                x, y, w, h = cv2.boundingRect(mask_contour)
                cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 0, 255), 3)  # drawing rectangle

    cv2.imshow("mask image", mask)  # Displaying mask image
    cv2.imshow("OG", img_color)  # Displaying webcam image
    cv2.imshow('What sensor sees', hstack)
    cv2.waitKey(1)

