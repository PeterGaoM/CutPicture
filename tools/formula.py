# -*- coding: utf-8 -*-
import cv2
from math import *
import time,math
import numpy as np
import pprint as pp
import os.path 
import tensorflow as tf
import os
import re
import shutil

def mass_per_fiber(width,flag):
	if flag == 0:
		mass= 1.54 * width * width * 15.5
	elif flag == 1:
		mass = 1.50 * width * width * 15.5
	elif flag == 2:
		mass = 1.52 * width * width * 15.5
	elif flag == 3:
		mass = 1.18 * width * width * 15.5
	elif flag == 4:
		mass = 1.13 * width * width * 15.5
	return mass

def cutPic(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
	gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

	# subtract the y-gradient from the x-gradient
	gradient = cv2.subtract(gradX, gradY)
	gradient = cv2.convertScaleAbs(gradient)

	# blur and threshold the image
	blurred = cv2.blur(gradient, (9, 9))
	(_, thresh) = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

	closed = thresh

	a = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = a[1]
	c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

	# compute the rotated bounding box of the largest contour
	rect = cv2.minAreaRect(c)
	box = np.int0(cv2.boxPoints(rect))

	# rotate
	pt1 = box[0]
	pt2 = box[1]
	pt3 = box[2]
	pt4 = box[3]
	withRect = math.sqrt((pt4[0] - pt1[0]) ** 2 + (pt4[1] - pt1[1]) ** 2)  # ���ο�Ŀ��
	heightRect = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) **2)
	angle = acos((pt4[0] - pt1[0]) / withRect) * (180 / math.pi)  # ���ο���ת�Ƕ�

	if pt4[1]>pt1[1]:
		angle=angle
	else:
		angle=-angle
		
	height = image.shape[0]
	width = image.shape[1]
	rotateMat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)  # ��angle�Ƕ���תͼ��
	heightNew = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))
	widthNew = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))
	rotateMat[0, 2] += (widthNew - width) / 2
	rotateMat[1, 2] += (heightNew - height) / 2
	imgRotation = cv2.warpAffine(image, rotateMat, (widthNew, heightNew), borderValue=(255, 255, 255))
	[[pt1[0]], [pt1[1]]] = np.dot(rotateMat, np.array([[pt1[0]], [pt1[1]], [1]]))
	[[pt3[0]], [pt3[1]]] = np.dot(rotateMat, np.array([[pt3[0]], [pt3[1]], [1]]))
	[[pt2[0]], [pt2[1]]] = np.dot(rotateMat, np.array([[pt2[0]], [pt2[1]], [1]]))
	[[pt4[0]], [pt4[1]]] = np.dot(rotateMat, np.array([[pt4[0]], [pt4[1]], [1]]))

	if pt2[1]>pt4[1]:
		pt2[1],pt4[1]=pt4[1],pt2[1]
	if pt1[0]>pt3[0]:
		pt1[0],pt3[0]=pt3[0],pt1[0]

	imgOut = imgRotation[int(pt2[1]):int(pt4[1]), int(pt1[0]):int(pt3[0])]
	return imgOut

def width(image,deta):
	temp=cutPic(image)
	temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
	final = []
	for pixel in range(255):
		ret, binary = cv2.threshold(temp, pixel, 255, cv2.THRESH_BINARY )
		[rows, cols] = binary.shape
		position = []
		if rows<cols:
			#���±ߵı�
			i = 0
			position_down = []
			while i<cols:
				temp_position = [i,rows]
				c = 0
				while c<rows:
					j = rows-c-1
					c=c+1
					if binary[j,i]==0:
						temp_position[0] = i
						temp_position[1] = j
						break
				i=i+deta
				position_down.append(temp_position)
			position_up = []
			i = 0
			#���ϱߵı�
			while i<cols:
				temp_position = [i,0]
				j = 0
				while j<rows:
					if binary[j,i]==0:
						temp_position[0] = i 
						temp_position[1] = j
						break
					j=j+1
				i=i+deta
				position_up.append(temp_position)
			position.append(position_up)
			position.append(position_down)
		else:	
			position_right = []
			i = 0
			#���ұߵı�
			while i<rows:
				temp_position = [cols, i]
				c = 0
				while c<cols:
					j = cols-c-1
					c=c+1
					if binary[i,j]==0:
						temp_position[0] = j  
						temp_position[1] = i
						break
				i=i+deta
				position_right.append(temp_position)
			position_left = []
			i = 0
			#����ߵı�
			while i<rows:
				temp_position = [0, i]
				j = 0
				while j<cols:
					if binary[i,j]==0:
						temp_position[0] = j  
						temp_position[1] = i
						break
					j=j+1
				i=i+deta
				position_left.append(temp_position)
			position.append(position_left)
			position.append(position_right)

		result = np.array(position)
		rest = result[1,:,:]-result[0,:,:]

		if cols>rows:
			rest = rest[:,1]
			m =np.where(rest>10)
			rest =rest[m]
			n = np.where(rest<rows*0.8)
			rest[n]
		else:
			rest = rest[:,0]
			m =np.where(rest>10)
			rest =rest[m]
			n = np.where(rest<cols*0.8)
			rest[n]
		final.append(np.mean(rest))
	return min(final)
	

