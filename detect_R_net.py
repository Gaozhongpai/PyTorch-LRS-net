from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from PIL import Image
import torch.utils.data as data
import os
import numpy as np
from PIL import Image, ImageDraw
from Pretrain_R_net import Net
#from R_net import Net

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir):
        self.img_path = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]

    def __getitem__(self, index):
        img = transforms.ToTensor()(Image.open(self.img_path[index]).convert('RGB'))
        return img, self.img_path[index]

    def __len__(self):
        return len(self.img_path)

def test(model, device, image_paths, images):
    model.eval()
    save_path = './output_sec_pre/'
    
    ind = 0
    images = images.to(device)
    output = model(images).double()
    output = output * 100

    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        draw.polygon([output[ind, 0], output[ind, 1], output[ind, 2], output[ind, 3], output[ind, 4], output[ind, 5], output[ind, 6], output[ind, 7]], outline=(0,255,0))	
        
        name = img_path.split('/')
        savepath = os.path.join(save_path, name[5])
        img.save(savepath)

        target_path = ''
        target_path = img_path.replace("secimages", "R-net-labels").replace(".jpg", ".txt")
        with open(target_path, 'w') as f:
            s = '{} {} {} {} {} {} {} {}\n'.format(output[ind, 0], output[ind, 1], output[ind, 2], output[ind, 3], 
                                                    output[ind, 4], output[ind, 5], output[ind, 6], output[ind, 7])
            f.write(s)
        ind += 1


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    dataset = DatasetFromFolder('./data/custom/secimages/train/')
    test_loader = torch.utils.data.DataLoader(dataset, **test_kwargs)
    model_path = './checkpoints/R_net_model/R_net_pretrain_100.pth'
    #model_path = './R_net_model/R_net_model_100.pth'
    model = Net().to(device)
    model.load_state_dict(torch.load(model_path))
    for batch_i, (input_imgs, img_paths) in enumerate(test_loader):
        test(model, device, img_paths, input_imgs)


if __name__ == '__main__':
    main()