# -*- coding:utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
import torchvision.models as models
from DataPrepare.config import config
from torch.nn import functional as F
import numpy as np
from utils.utils import Utils

device = config.device
feature_extract = config.feature_extract
input_size = config.input_size


class LeNet(nn.Module):
    def __init__(self, num_class=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10, num_class)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class CNNModel(object):
    def __init__(self, num_classes=2, model_name="inception", model_path="", feature_extract=True):
        if model_name == "inception":
            self.model = models.inception_v3()
            self.model.load_state_dict(torch.load(model_path))
            self.set_parameter_requires_grad(self.model, feature_extract)
            # Handle the auxilary net
            num_ftrs = self.model.AuxLogits.fc.in_features
            self.model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
            # input_size = 299
        elif model_name == "resnet18":
            self.model = models.resnet18()
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.set_parameter_requires_grad(self.model, feature_extract)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
            # input_size = 224
        elif model_name == "resnet34":
            self.model = models.resnet34()
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.set_parameter_requires_grad(self.model, feature_extract)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name == "resnet50":
            self.model = models.resnet50()
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.set_parameter_requires_grad(self.model, feature_extract)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name == "resnet101":
            self.model = models.resnet101()
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.set_parameter_requires_grad(self.model, feature_extract)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name == "resnet152":
            self.model = models.resnet152()
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.set_parameter_requires_grad(self.model, feature_extract)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name == "mobilenet":
            self.model = models.mobilenet_v2()
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.set_parameter_requires_grad(self.model, feature_extract)
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.model.last_channel, num_classes),
            )
        elif model_name == "shufflenet":
            self.model = models.shufflenet_v2_x1_0()
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.set_parameter_requires_grad(self.model, feature_extract)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name == "squeezenet":
            self.model = models.squeezenet1_1()
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.set_parameter_requires_grad(self.model, feature_extract)
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Conv2d(512, num_classes, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
        else:
            raise ValueError("Your pretrain model name is wrong!")

    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False


class DrownModel(object):
    def __init__(self, class_nums, pre_train_name, pre_train_path, model_path):
        self.model = CNNModel(class_nums, pre_train_name, pre_train_path, feature_extract).model.to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.image_normalize = Utils.image_normalize

    def predict(self, img):
        img_tensor_list = []
        img_tensor = self.image_normalize(img, size=input_size)
        img_tensor_list.append(torch.unsqueeze(img_tensor, 0))
        if len(img_tensor_list) > 0:
            input_tensor = torch.cat(tuple(img_tensor_list), dim=0)
            res_array = self.predict_image(input_tensor)
            return res_array

    def predict_image(self, image_batch_tensor):
        self.model.eval()
        image_batch_tensor = image_batch_tensor.cuda()
        outputs = self.model(image_batch_tensor)
        outputs_tensor = outputs.data
        m_softmax = nn.Softmax(dim=1)
        outputs_tensor = m_softmax(outputs_tensor).to("cpu")
        return np.asarray(outputs_tensor)


if __name__ == "__main__":
    # model = LstmModel()
    # test_tensor = torch.randn((32, 30, 512))
    # print(test_tensor.size())
    # tt = model(test_tensor)
    # print(tt)
    a = 1
