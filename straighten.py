import numpy as np
import cv2
import os
from PIL import Image, ImageDraw

def transform(img, point_set_0, point_set_1):
    mat = cv2.getPerspectiveTransform(point_set_0, point_set_1)
    lic = cv2.warpPerspective(img, mat, (96, 96))

    return lic


point_set_1 = np.array([[0, 0], [96, 0], [96, 96], [0, 96]], dtype=np.float32)

source_path = './data/custom/secimages/train/'
save_path = './data/custom/straightened/train/'
image_list=os.listdir(source_path)
for file in image_list:
    path = source_path + file
    img = cv2.imread(path)
    label_path = path.replace('secimages', 'R-net-labels').replace(".jpg", ".txt")
    label = np.loadtxt(label_path).reshape(4, 2).astype(np.float32)
    re = transform(img, label, point_set_1)
    save = save_path + file
    cv2.imwrite(save, re)