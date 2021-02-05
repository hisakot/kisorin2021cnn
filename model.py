
import torch
import torch.nn as nn
import torchvision.models as models

class AlexNet(nn.Module):
    def __init__(self, pretrained=False, out_classes=1000):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.dropout = nn.Dropout(p=0.5, inplace=False)
        self.linear1 = nn.Linear(in_features=256 * 6 * 6, out_features=4096)
        self.linear2 = nn.Linear(4096, 4096)
        self.linear2 = nn.Linear(4096, out_classes)

    def foward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = x.view(x.size(0), 256 * 6 * 6)

        x = self.dropout(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)

        return x

class AlexNet_Model(nn.Module):
    def __init__(self, pretrained=False, out_classes=1000):
        super(AlexNet_Model, self).__init__()
        # pre-trained on ImageNet, display progress bar
        self.alexnet = models.alexnet(pretrained=pretrained, progress=False)
        self.fc = nn.Linear(1000, out_classes)

    def forward(self, x):
        x = self.alexnet(x)
        x = self.fc(x)
        
        return x
