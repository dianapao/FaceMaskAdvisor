from torch.utils.data import DataLoader
import torchvision
from libs.FaceMaskDataset import FaceMaskDataset
from torchvision.transforms import Compose, Resize, CenterCrop
import torch
from torch import nn, optim

transform = Compose([
    Resize(256),
    CenterCrop(224)
])

dataset = FaceMaskDataset('./data/dataset/annotations.csv', './data/dataset/images', transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
model = nn.Sequential(
    torchvision.models.googlenet(pretrained=True),
    nn.Linear(1000, 1),
    nn.Sigmoid()
)
print(model)

loss_fn = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=0.0005, momentum=0.9)

for i in range(1, 16):
    print(f'Epoch {i:>2}:')
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        y = y.reshape(-1, 1).float()
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss, current = loss.item(), batch * len(X)
        print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')

torch.save(model.state_dict(), 'model.pth')