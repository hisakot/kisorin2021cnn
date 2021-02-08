import cv2
import numpy as np
from tqdm import tqdm

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms

import common
import dataset
import model


def trainer(train, model, optimizer, lossfunc):
    print("---------- Start Training ---------")
    trainloader = torch.utils.data.DataLoader(
            train, batch_size=1, shuffle=True, num_workers=3)

    try:
        with tqdm(trainloader, ncols=100) as pbar:
            train_loss = 0.0
            for images, labels in pbar:
                images, labels = Variable(images), Variable(labels)
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = lossfunc(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

        return train_loss

    except ValueError:
        pass

def tester(test, model):
    print("---------- Start Testing ---------")
    testloader = torch.utils.data.DataLoader(
            test, batch_size=1, shuffle=False, num_workers=3)

    try:
        test_loss = 0.0
        with tqdm(testloader, ncols=100) as pbar:
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = lossfunc(outputs, labels)
                test_loss += loss.item()

        return test_loss

    except ValueError:
        pass

if __name__ == '__main__':
    print("---------- Loading Data ----------")
    datas = dataset.setup_data()
    print("---------- Finished loading Data ----------")

    # split train and test
    train_size = int(round(datas.length * 0.8))
    test_size = datas.length - train_size
    train, test = torch.utils.data.random_split(datas, [train_size, test_size])

    # set up model
    model = model.AlexNet(pretrained=False, out_classes=5)
#     model = model.AlexNet_Model(pretrained=False, out_classes=5) # x, y

    # set up GPU
    model, device = common.setup_device(model)

    # optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    lossfunc = nn.CrossEntropyLoss()
#     lossfunc = nn.MSELoss()

    # tensorboard
    writer = SummaryWriter(log_dir="./logs")

    # main
    loss_list = list()
    early_stopping = [np.inf, 3, 0]
    for epoch in range(100):
        try:
            # train
            train_loss = trainer(train, model, optimizer, lossfunc)
            loss_list.append(train_loss)

            # test
            with torch.no_grad():
                test_loss = tester(test, model)

            # show loss and accuracy
            print("%d : train_loss : %.3f" % (epoch + 1, train_loss))
            print("%d : test_loss : %.3f" % (epoch + 1, test_loss))
            torch.save({
                "epoch" : epoch + 1,
                "model_state_dict" : model.state_dict(),
                "optimizer_state_dict" : optimizer.state_dict(),
                "loss" : loss_list
                }, "./models/" + str(epoch + 1))
            
            # early stoping
#             if test_loss < early_stopping[0]:
#                 early_stopping[0] = test_loss
#                 early_stopping[-1] = 0
#                 torch.save({
#                     "epoch" : epoch + 1,
#                     "model_state_dict" : model.state_dict(),
#                     "optimizer_state_dict" : optimizer.state_dict(),
#                     "loss" : loss_list
#                     }, "./models/" + str(epoch + 1))
#             else:
#                 early_stopping[-1] += 1
#                 if early_stopping[-1] == early_stopping[1]:
#                     break

            # tensorboard
            writer.add_scalar("Train Loss", train_loss, epoch)
            writer.add_scalar("Test Loss", test_loss, epoch)

        except ValueError:
            continue
