from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from PIL import Image
import torch.utils.data as data
import os
import numpy as np


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir):
        self.img_path = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]
        self.label_files = [
            path.replace("secimages", "seclabels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_path
        ]

    def __getitem__(self, index):
        img = transforms.ToTensor()(Image.open(self.img_path[index]).convert('RGB'))
        _, h, w = img.shape
        label_path = self.label_files[index].rstrip()
        targets = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 9))
        targets = targets[0]
        targets = targets[1:]
        targets = targets / h

        return img, targets

    def __len__(self):
        return len(self.img_path)


class Net(nn.Module):
    def __init__(self,is_train=False, use_cuda=True):
        super(Net, self).__init__()
        self.is_train = is_train
        self.use_cuda = use_cuda
        # backend
        self.base = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'mixnet_s', pretrained=True)
        self.linear = nn.Linear(1536, 8)

    def forward(self, x):
        # backend
        x = self.base.conv_stem(x)
        x = self.base.bn1(x)
        x = self.base.act1(x)
        x = self.base.blocks(x)
        x = self.base.conv_head(x)
        x = self.base.bn2(x)
        x = self.base.act2(x)
        x = self.base.global_pool(x)
        x = x.view(x.size(0), -1)
        # detection
        landmark = torch.sigmoid(self.linear(x))
        return landmark


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (img, target) in enumerate(train_loader):
        img, target = img.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(img).double()

        edge12 = torch.norm(target[:, 0:2] - target[:, 2:4])
        edge12_pred = torch.norm(output[:, 0:2] - output[:, 2:4])
        loss_edge12 = nn.MSELoss()(edge12, edge12_pred)

        edge23 = torch.norm(target[:, 2:4] - target[:, 4:6])
        edge23_pred = torch.norm(output[:, 2:4] - output[:, 4:6])
        loss_edge23 = nn.MSELoss()(edge23, edge23_pred)

        edge34 = torch.norm(target[:, 4:6] - target[:, 6:8])
        edge34_pred = torch.norm(output[:, 4:6] - output[:, 6:8])
        loss_edge34 = nn.MSELoss()(edge34, edge34_pred)

        edge41 = torch.norm(target[:, 6:8] - target[:, 0:2])
        edge41_pred = torch.norm(output[:, 6:8] - output[:, 0:2])
        loss_edge41 = nn.MSELoss()(edge41, edge41_pred)

        loss_x1 = nn.L1Loss()(target[:,0], output[:,0])
        loss_y1 = nn.L1Loss()(target[:,1], output[:,1])
        loss_x2 = nn.L1Loss()(target[:,2], output[:,2])
        loss_y2 = nn.L1Loss()(target[:,3], output[:,3])
        loss_x3 = nn.L1Loss()(target[:,4], output[:,4])
        loss_y3 = nn.L1Loss()(target[:,5], output[:,5])
        loss_x4 = nn.L1Loss()(target[:,6], output[:,6])
        loss_y4 = nn.L1Loss()(target[:,7], output[:,7])

        loss = loss_x1 + loss_y1 + loss_x2 + loss_y2 + loss_x3 + loss_y3 + loss_x4 + loss_y4 + \
                    loss_edge12 + loss_edge23 + loss_edge34 + loss_edge41
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\nloss_x1: {:.6f}\tloss_y1: {:.6f}\tloss_x2: {:.6f}\tloss_y2: {:.6f}\
            \nloss_x3: {:.6f}\tloss_y3: {:.6f}\tloss_x4: {:.6f}\tloss_y4: {:.6f}'.format(
                epoch, batch_idx * len(img), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), loss_x1.item(), loss_y1.item(), loss_x2.item(), loss_y2.item()\
                , loss_x3.item(), loss_y3.item(), loss_x4.item(), loss_y4.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for img, target in test_loader:
            img, target = img.to(device), target.to(device)
            output = model(img).double()
            loss_x1 = nn.L1Loss()(target[:,0], output[:,0])
            loss_y1 = nn.L1Loss()(target[:,1], output[:,1])
            loss_x2 = nn.L1Loss()(target[:,2], output[:,2])
            loss_y2 = nn.L1Loss()(target[:,3], output[:,3])
            loss_x3 = nn.L1Loss()(target[:,4], output[:,4])
            loss_y3 = nn.L1Loss()(target[:,5], output[:,5])
            loss_x4 = nn.L1Loss()(target[:,6], output[:,6])
            loss_y4 = nn.L1Loss()(target[:,7], output[:,7])

            test_loss += (loss_x1 + loss_y1 + loss_x2 + loss_y2 + loss_x3 + loss_y3 + loss_x4 + loss_y4)
            print('\nTrain Epoch: \tLoss: {:.6f}\nloss_x1: {:.6f}\tloss_y1: {:.6f}\tloss_x2: {:.6f}\tloss_y2: {:.6f}\
            \nloss_x3: {:.6f}\tloss_y3: {:.6f}\tloss_x4: {:.6f}\tloss_y4: {:.6f}'.format(
                test_loss.item(), loss_x1.item(), loss_y1.item(), loss_x2.item(), loss_y2.item()\
                , loss_x3.item(), loss_y3.item(), loss_x4.item(), loss_y4.item()))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.92, metavar='M',
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

    dataset1 = DatasetFromFolder('./data/custom/secimages/train')
    dataset2 = DatasetFromFolder('./data/custom/secimages/valid')
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        #test(model, device, test_loader)
        scheduler.step()
        if epoch % 20 == 0:
            torch.save(model.state_dict(), f"R_net_model/R_net_pretrain_%d.pth" % epoch)


if __name__ == '__main__':
    main()