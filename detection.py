from Classes.Prediction import crossdetect, Prediction
from tools.tools import *
from numpy import *
import time


Model2 = './Model/model2/ckpt'
Model3 = './Model/model3/freeze_graph.pb'
RESIZED_IMAGE = (100, 100)


#将语义分割的图片与目标检测到的交叉点整合到一张图片上面
def group(image, det_boxes):
    for i in det_boxes[0]:
        if i[2]*i[3] > image.shape[0]*image.shape[1]/5:
            continue

        image = cv2.circle(image,
                       (int(i[0] + i[2] / 2), int(i[1] + i[3] / 2)),
                        int(i[2] / 2),
                        (0, 255, 0),
                        -1)
    return image


def segmentation(image):
    new_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #new_image = (0.5 * cos((mat(image) / 255.0 + 1) * pi) + 0.5) * 255
    #new_image = -255 / (x2 - x1) * (mat(image) - x2)
    new_image = np.array(new_image, dtype=image.dtype)

    gradX = cv2.Sobel(new_image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(new_image, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    # blur and threshold the image
    blurred = cv2.blur(gradient, (15, 15))
    (_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
    g = np.zeros(new_image.shape, new_image.dtype)
    b = np.zeros(new_image.shape, new_image.dtype)
    img = cv2.merge([b, g, thresh])
    return img


def main(path, folder):
    predict2 = crossdetect(Model2)
    predict3 = Prediction(Model3, 'input:0', 'output:0')

    for name in os.listdir(path):
        start = time.time()
        image = cv2.imread(path +'/'+ name)
        image = cv2.resize(image, (1000, 1000))

        preds2 = predict2.predict(image)

        img = group(segmentation(image), preds2)

        TrainPicture(image,
                     img,
                     folder,
                     name,
                     predict3)
        end = time.time()
        print(name, end - start, 's')



if __name__ == '__main__':
    dir = '../'
    for folder in os.listdir(dir):
        if 'test' in folder:
            path = dir + folder+'/Init_Fibre'
            main(path, dir + folder)