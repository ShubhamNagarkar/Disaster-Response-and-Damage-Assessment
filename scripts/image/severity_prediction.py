import numpy as np
import torch
import argparse
import sys
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchsummary import summary
from PIL import Image
import os
from collections import Counter
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.metrics import confusion_matrix

from ResNet50 import Resnet50


DEVICE = ("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-4
PATIENCE = 5
N_WORKERS = 8
IMAGE_SIZE = (224, 224)


REL = os.getcwd()
PATH_TRAIN = REL + '/dataset/images/Train/'
PATH_TEST =  REL + '/dataset/images/Test/'
PATH_VAL = REL + '/dataset/images/Val/'


class Severity(torch.nn.Module):
    def __init__(self):
        super(Severity,self).__init__()

        self.resnet = torchvision.models.resnet50(pretrained=True)
        modules = list(self.resnet.children())[:-1]
        self.resnet = torch.nn.Sequential(*modules)
    
        for params in self.resnet.parameters():
            params.requires_grad = False
        
        self.head = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=2048, out_features=512, bias=True),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(512, 3, bias=True))


    def forward(self, x):
        feat = self.resnet(x)
        output = self.head(feat)

        return output


class FineTune(torch.nn.Module):
    def __init__(self, loaded_model):
        super(FineTune,self).__init__()

        self.backbone = loaded_model
    
        for params in self.backbone.parameters():
            params.requires_grad = False
        

        self.head = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=2048, out_features=512, bias=True),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(512, 3, bias=True))


    def forward(self, x):
        feat = self.backbone(x)
        output = self.head(feat)

        return output


if __name__ == "__main__":

    model = Resnet50()
    model.to(DEVICE)
