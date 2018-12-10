import os
import cv2

From = ''
Dest = ''

def main(path, destination, folder):
    for name in os.listdir(path):
        image = cv2.imread(path + '/' + name)
        name = str(name[:len(name) - 4])
        cv2.imwrite(destination + '/'+folder+'__'+name + '.jpg', image)
    print('Finished!')


if __name__ == '__main__':
    dir = '../'
    for folder in os.listdir(dir):
        if From in folder:
            path = dir + folder + '/Sep_Fibre'
            main(path, Dest, folder)