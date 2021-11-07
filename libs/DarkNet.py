from torch import nn

class DarkNet(nn.Module):
    def __init__(self) -> None:
        super(DarkNet, self).__init__()
        self.model_stack = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.MaxPool2d(2,2),

            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            nn.MaxPool2d(2,2),

            nn.Conv2d(256, 512, 3),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            nn.Conv2d(256, 512, 3),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            nn.Conv2d(256, 512, 3),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            nn.MaxPool2d(2,2),

            nn.Conv2d(512, 1024, 3),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            nn.Conv2d(512, 1024, 3),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            nn.Conv2d(512, 1024, 3),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            nn.Conv2d(1024, 1000, 1),
            nn.BatchNorm2d(1000),
            nn.LeakyReLU(0.1),

            nn.AvgPool2d(1000),
            nn.Softmax()
        )
    
    def forward(self, x):
        return self.model_stack(x)