from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import cv2
from PIL import Image, ImageDraw

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from train_R_net import Net
import segmentation_models_pytorch as smp

def transform(img, point_set_0, point_set_1):
    mat = cv2.getPerspectiveTransform(point_set_0, point_set_1)
    lic = cv2.warpPerspective(img, mat, (96, 96))
    return lic

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_path = './data/test/label5_00562.jpg'
    output_path = './test/'
    os.makedirs(output_path, exist_ok=True)

    img_size_L = 416
    img_size_R = 100
    img_size_S = 96
    crop_size = (img_size_R, img_size_R)

    L_net = './checkpoints/L_net_model/L_net_model.pth'
    R_net = './checkpoints/R_net_model/R_net_model.pth'
    S_net = './checkpoints/S_net_model/S_net_model.pth'

    model_L = Darknet("config/yolov3-custom.cfg", img_size=416).to(device)
    model_L.load_state_dict(torch.load(L_net))
    model_L.eval()

    model_R = Net().to(device)
    model_R.load_state_dict(torch.load(R_net))
    model_R.eval()

    model_S = smp.Unet(
        encoder_name="timm-efficientnet-b1", 
        encoder_weights="imagenet",
        in_channels=3,      
        classes=2,        
    )
    model_S = model_S.to(device)
    model_S.load_state_dict(torch.load(S_net))
    model_S.eval()

    input_img_L = Image.open(input_path).convert('RGB')
    input_img_R = cv2.imread(input_path)
    img = transforms.ToTensor()(input_img_L)
    img, _ = pad_to_square(img, 0)
    img = resize(img, img_size_L)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    img = Variable(img.type(Tensor)).unsqueeze(0)

    with torch.no_grad():
        detections = model_L(img)            
        detections = non_max_suppression_xy(detections, conf_thres=0.8, nms_thres=0.4)

    for i in range(len(detections)):
        detection = detections[i]
        detection = rescale_boxes_xy(detection, img_size_L, np.array(input_img_L).shape[:2])
        
        x_max = min(int(detection[:,10])+10, 640)
        x_min = max(int(detection[:,8])-10, 0)
        y_max = min(int(detection[:,11])+10, 360)
        y_min = max(int(detection[:,9])-10, 0)
        w = x_max-x_min
        h = y_max-y_min
        input_img_R = input_img_R[y_min:y_max, x_min:x_max]
        input_img_R = cv2.resize(input_img_R, crop_size, interpolation = cv2.INTER_CUBIC)
        img_R = transforms.ToTensor()(input_img_R).unsqueeze(0).to(device)
        output_R = model_R(img_R).double() * img_size_R
        output_R = output_R.cpu().detach().numpy().reshape(4,2).astype(np.float32)
        point_set_1 = np.array([[0, 0], [img_size_S, 0], [img_size_S, img_size_S], [0, img_size_S]], dtype=np.float32)
        input_img_S = transform(input_img_R, output_R, point_set_1)
        img_S = transforms.ToTensor()(input_img_S).unsqueeze(0).to(device)
        output_S = model_S(img_S).double() * 100
        index = torch.max(output_S[0], dim = 0)[1]
        result = index[...,None].repeat(1, 1, 3).cpu().numpy()*255

        savepath = os.path.join(output_path, str(i+1)+'.jpg')
        cv2.imwrite(savepath, result)



