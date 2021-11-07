import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import googlenet
from torchvision.transforms import Compose, Resize, CenterCrop

from libs.FaceMaskDataset import FaceMaskDataset

if __name__ == '__main__':

    model = nn.Sequential(
        googlenet(pretrained=True),
        nn.Linear(1000, 1),
        nn.Sigmoid()
    )
    model.load_state_dict(torch.load('./model.pth'))
    model.eval()

    transforms = Compose([Resize(256), CenterCrop(224)])
    with torch.no_grad():
        dataset = FaceMaskDataset('./data/dataset/annotations.csv', './data/dataset/images', transforms)
        dataloader = DataLoader(dataset, 64, True)

        acc = 0
        for X, y in dataloader:
            pred = model.forward(X)
            pred = (1.0 * (pred > 0.5)).ravel()
            acc += torch.abs(pred - y).sum()
        print(f'Accuracy: {len(dataset) - acc}/{len(dataset)} = {1. - acc/len(dataset)}')