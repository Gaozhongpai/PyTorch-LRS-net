import cv2
import os
import numpy as np

source_img_path = './data/custom/images/train/l4/'
target_path = "./data/custom/secimages/train/"

img_size = 100  

crop_size = (img_size, img_size)
 
image_list=os.listdir(source_img_path)

for file in image_list:
    path = source_img_path + file
    lpath = path.replace("images", "labels").replace(".jpg", ".txt")
    image_source=cv2.imread(path)
    label = np.loadtxt(lpath).reshape(-1, 9)
    for i in range(len(label)):
        tmp_label = label[i]
        tmp = tmp_label[1:8:2]
        x_max = min(int(np.max(tmp))+5, 640)
        x_min = max(int(np.min(tmp))-5, 0)
        tmp = tmp_label[2:9:2]
        y_max = min(int(np.max(tmp))+5, 360)
        y_min = max(int(np.min(tmp))-5, 0)
        w = x_max-x_min
        h = y_max-y_min
        cropped = image_source[y_min:y_max, x_min:x_max]
        image = cv2.resize(cropped, crop_size, interpolation = cv2.INTER_CUBIC)
        (filename, extension) = os.path.splitext(file)
        filename = filename + '_' + str(i+1) + extension
        cv2.imwrite(target_path+filename, image)
        tlpath = target_path + filename
        tlpath = tlpath.replace("secimages", "seclabels").replace(".jpg", ".txt")
        tmp_label[1] = (tmp_label[1]-x_min) * img_size / w
        tmp_label[2] = (tmp_label[2]-y_min) * img_size / h
        tmp_label[3] = (tmp_label[3]-x_min) * img_size / w
        tmp_label[4] = (tmp_label[4]-y_min) * img_size / h
        tmp_label[5] = (tmp_label[5]-x_min) * img_size / w
        tmp_label[6] = (tmp_label[6]-y_min) * img_size / h
        tmp_label[7] = (tmp_label[7]-x_min) * img_size / w
        tmp_label[8] = (tmp_label[8]-y_min) * img_size / h
        with open(tlpath, 'w') as f:
            s = '0 {} {} {} {} {} {} {} {}\n'.format(tmp_label[1], tmp_label[2], tmp_label[3], tmp_label[4], 
                                                    tmp_label[5], tmp_label[6], tmp_label[7], tmp_label[8])
            f.write(s)
    
