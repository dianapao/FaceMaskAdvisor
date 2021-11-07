import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import googlenet
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, CenterCrop
import argparse

from libs.FaceMaskDataset import FaceMaskDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face mask detection script')
    parser.add_argument('--image',
        type=str,
        required=True,
        help='image path'
    )
    args = parser.parse_args()

    model = nn.Sequential(
        googlenet(pretrained=True),
        nn.Linear(1000, 1),
        nn.Sigmoid()
    )
    model.load_state_dict(torch.load('./model.pth'))
    model.eval()

    transforms = Compose([Resize(256), CenterCrop(224)])
    with torch.no_grad():
        img = read_image(args.image)
        img = transforms(img).float().unsqueeze(0)
        pred = model.forward(img)
        if pred.item() > 0.5:
            print('Felicidades! sabes usar un cubrebocas')
        else:
            print('Lo estas usando mal >:c')
        print(f'Probabilidad de que se usa bien: {pred.item()*100:>3.2f}%')
