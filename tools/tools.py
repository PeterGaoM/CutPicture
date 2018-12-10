import scipy.io as sio
import tensorflow as tf
import cv2
import numpy as np
from math import *
import math
from scipy import misc
import os

label_colours = [[0, 0, 0], [128, 0, 0], [0, 128, 0]]
RESIZED_IMAGE = (100, 100)


def read_labelcolours(matfn):
    mat = sio.loadmat(matfn)
    color_table = mat['colors']
    shape = color_table.shape
    color_list = [tuple(color_table[i]) for i in range(shape[0])]

    return color_list


def decode_labels(mask, img_shape, num_classes):
    color_table = label_colours

    color_mat = tf.constant(color_table, dtype=tf.float32)
    onehot_output = tf.one_hot(mask, depth=num_classes)
    onehot_output = tf.reshape(onehot_output, (-1, num_classes))
    pred = tf.matmul(onehot_output, color_mat)
    pred = tf.reshape(pred, (1, img_shape[0], img_shape[1], 3), name='output')
    
    return pred


def prepare_label(input_batch, new_size, num_classes, one_hot=True):
    with tf.name_scope('label_encode'):
        input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size) # as labels are integer numbers, need to use NN interp.
        input_batch = tf.squeeze(input_batch, squeeze_dims=[3]) # reducing the channel dimension.
        if one_hot:
            input_batch = tf.one_hot(input_batch, depth=num_classes)
            
    return input_batch


def extension(img, size):
	#拓宽图片的边缘，
	#输入：用于拓宽的图片（灰度图），以及需要达到的大小
	#输出：拓宽之后的图片
	COLOR = [115, 115, 115]
	width, height, channels = img.shape
	detaX = size - width
	detaY = size - height
	if width > size:
		width = size
		detaX = 0
	if height > size:
		height = size
		detaY = 0

	res = cv2.resize(img, (height, width), interpolation=cv2.INTER_AREA)
	top = int(detaX / 2)
	left = int(detaY / 2)
	final = cv2.copyMakeBorder(res, top, detaX - top, left, detaY - left,cv2.BORDER_CONSTANT, value=[115,115,115])
		#  cv2.BORDER_REPLICATE)  # cv2.BORDER_CONSTANT, value=COLOR)
	return final


def contactOrNot(contour1, contour2, r):
    image1 = np.zeros(r.shape)
    image2 = np.zeros(r.shape)
    cv2.drawContours(image1, [contour1], 0, 255, -1)
    cv2.drawContours(image2, [contour2], 0, 255, -1)
    image = image1 + image2
    image = np.clip(image, 0, 255)  # 归一化也行
    image = np.array(image, np.uint8)
    _, num, __ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #print(image[image>50])
    if len(num) == 1:
        return True
    else:
        return False


def deletSmallArea(img, are):
    image, contours, hierarch = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = contours
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < are:
            cv2.drawContours(image, [contours[i]], 0, 0, -1)
        image, contour, hierarch = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return image, contour


def fullArea(img):
	#用白色填充轮廓中空余的部分
    #输入：二值化图片
    #输出：填充之后的二值图
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
	g = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
	return g


def CutContour(image, contour):
	#根据识别的轮廓对图片进行切割并旋转
	#输入：{image:原图， contour: 轮廓}
	#输出：对应轮廓的图片
	# compute the rotated bounding box of the largest contour
	rect = cv2.minAreaRect(contour)

	box = np.int0(cv2.boxPoints(rect))
	print(box)
	# rotate
	pt1 = box[0]
	pt2 = box[1]
	pt3 = box[2]
	pt4 = box[3]
	withRect = math.sqrt((pt4[0] - pt1[0]) ** 2 + (pt4[1] - pt1[1]) ** 2)
	heightRect = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
	angle = acos((pt4[0] - pt1[0]) / withRect) * (180 / math.pi)

	if pt4[1] > pt1[1]:
		angle = angle
	else:
		angle = -angle

	height = image.shape[0]
	width = image.shape[1]
	rotateMat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
	heightNew = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))
	widthNew = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))
	rotateMat[0, 2] += (widthNew - width) / 2
	rotateMat[1, 2] += (heightNew - height) / 2
	imgRotation = cv2.warpAffine(image, rotateMat, (widthNew, heightNew), borderValue=(255, 255, 255))
	[[pt1[0]], [pt1[1]]] = np.dot(rotateMat, np.array([[pt1[0]], [pt1[1]], [1]]))
	[[pt3[0]], [pt3[1]]] = np.dot(rotateMat, np.array([[pt3[0]], [pt3[1]], [1]]))
	[[pt2[0]], [pt2[1]]] = np.dot(rotateMat, np.array([[pt2[0]], [pt2[1]], [1]]))
	[[pt4[0]], [pt4[1]]] = np.dot(rotateMat, np.array([[pt4[0]], [pt4[1]], [1]]))

	if pt2[1] > pt4[1]:
		pt2[1], pt4[1] = pt4[1], pt2[1]
	if pt1[0] > pt3[0]:
		pt1[0], pt3[0] = pt3[0], pt1[0]

	imgOut = imgRotation[int(abs(pt2[1])):int(abs(pt4[1])), int(abs(pt1[0])):int(abs(pt3[0]))]


	return extension(imgOut, 500)


def TrainPicture(img, img2, path, filename, predict3):
	# 对语义分割的图片进行切割处理，将语义分割的图片以及切割的纤维图片放在output下
	# 在output下每张图片对应一个文件夹存放独立的纤维以及cross_num文件夹，
	#在cross_num文件夹分别存放每个交叉整体以及交叉的单跟纤维
	#输入：{img：原图， img2:语义分割图， filename:原图名称}
	#输出：无，

	#创建图片文件夹，保存识别图片
	name = path + '/Sep_Fibre/' + str(filename[:len(filename) - 4])
	b, g, r = cv2.split(img2)

	r = r.astype(np.uint8)
	image_r, contour_r = deletSmallArea(r, 5000)

	print(len(contour_r))
	for x in range(len(contour_r)):
		image = CutContour(img, contour_r[x])
		res = predict3.predict(image, RESIZED_IMAGE)

		if res[0][1] > 0.2:
			cv2.imwrite(name +'-'+ str(x) + '.jpg', image)


