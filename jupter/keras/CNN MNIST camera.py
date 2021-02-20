#import keras
import cv2 as cv
import numpy as np

#model = keras.models.load_model('MNIST_model.h5')
#model.summary()

cap = cv.VideoCapture(0)

while True:
	_, img = cap.read()
	img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	_, mask = cv.threshold(img_gray, 80, 255, cv.THRESH_BINARY_INV)
	mask = cv.dilate(mask, None, iterations=3)
	contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	cv.drawContours(img, contours, -1, (0, 0, 255), 2)

	for i, contour in enumerate(contours):
		x, y, w, h = cv.boundingRect(contour)
		cx, cy = x + w/2, y + h/2
		r = max(w,h) / 2 * 1.5
		x0, y0, x1, y1 = int(max(cx-r,0)), int(max(cy-r,0)), int(cx+r), int(cy+r)
		roi = img_gray[y0:y1,x0:x1]
		roi_inv = cv.bitwise_not(roi)
		if i < 10: cv.imshow(str(i), cv.resize(roi_inv, (280, 280)))
		cv.rectangle(img, (x0,y0), (x1,y1), (0,255,0), 2 )

	cv.imshow('img', img)
	cv.imshow('img_gray', img_gray)
	cv.imshow('mask', mask)
	if cv.waitKey(1) != -1: break