import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights, \
    EfficientNet_B3_Weights, EfficientNet_B4_Weights, EfficientNet_B5_Weights, EfficientNet_B6_Weights, \
    EfficientNet_B7_Weights, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights, \
    EfficientNet_V2_M_Weights, EfficientNet_V2_S_Weights, EfficientNet_V2_L_Weights
import os
import torch
from src import cct_14_7x2_384, cct_14_7x2_224, cct_7_7x2_224_sine
from torchsummary import summary

'''
Reference for Compact Convolutional Transformers
https://github.com/SHI-Labs/Compact-Transformers
'''

class ImageClassificationModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.num_classes = cfg["model"]["num_classes"]
        self.n_nodes = cfg["model"]["n_nodes"]
        self.dropout = cfg["model"]["dropout"]
        self.freeze_layers = cfg["model"].get("freeze_layers", 1.0) > 0.0
        self.image_size = cfg["data"]["size"]
        self.pretrained = cfg["model"].get("pretrained", 1.0) > 0.0
        self.model = self.get_model(cfg["model"]["name_pretrained_model"])
        summary(self.model, torch.zeros(1, 3, self.image_size, self.image_size))

    # connetto i vari strati definendo la funzione forward
    def forward(self, x):
        x = self.model(x)
        return x

    def classifier_head(self, output_dim):
        return nn.Sequential(nn.Flatten(),
                             nn.Linear(output_dim, self.n_nodes),
                             nn.ReLU(),
                             nn.Dropout(self.dropout),
                             nn.Linear(self.n_nodes, self.num_classes))

    def freeze_layers_base_model(self, model):
        for name, param in model.named_parameters():
            param.requires_grad = False

    def get_model(self, name_pretrained_model):
        if name_pretrained_model == 'resnet18':
            base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            base_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

            if self.freeze_layers:
                print("Freeze layers of pretrained model")
                self.freeze_layers_base_model(base_model)

            base_model.fc = self.classifier_head(512)
        elif name_pretrained_model == 'resnet34':
            base_model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
            base_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

            if self.freeze_layers:
                print("Freeze layers of pretrained model")
                self.freeze_layers_base_model(base_model)

            base_model.fc = self.classifier_head(512)
        elif name_pretrained_model == 'resnet50':
            base_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            base_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

            if self.freeze_layers:
                print("Freeze layers of pretrained model")
                self.freeze_layers_base_model(base_model)

            base_model.fc = self.classifier_head(2048)
        elif name_pretrained_model == 'resnet101':
            base_model = models.resnet101(weights=ResNet101_Weights.DEFAULT)
            base_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

            if self.freeze_layers:
                print("Freeze layers of pretrained model")
                self.freeze_layers_base_model(base_model)

            base_model.fc = self.classifier_head(2048)
        elif name_pretrained_model == 'resnet152':
            base_model = models.resnet152(weights=ResNet152_Weights.DEFAULT)
            base_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

            if self.freeze_layers:
                print("Freeze layers of pretrained model")
                self.freeze_layers_base_model(base_model)

            base_model.fc = self.classifier_head(2048)
        elif name_pretrained_model == 'efficientnet_b0':
            base_model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            base_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

            if self.freeze_layers:
                print("Freeze layers of pretrained model")
                self.freeze_layers_base_model(base_model)

            base_model.classifier = self.classifier_head(1280)
        elif name_pretrained_model == 'efficientnet_b1':
            base_model = models.efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
            base_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

            if self.freeze_layers:
                print("Freeze layers of pretrained model")
                self.freeze_layers_base_model(base_model)

            base_model.classifier = self.classifier_head(1280)
        elif name_pretrained_model == 'efficientnet_b2':
            base_model = models.efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
            base_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

            if self.freeze_layers:
                print("Freeze layers of pretrained model")
                self.freeze_layers_base_model(base_model)

            base_model.classifier = self.classifier_head(1408)
        elif name_pretrained_model == 'efficientnet_b3':
            base_model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
            base_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

            if self.freeze_layers:
                print("Freeze layers of pretrained model")
                self.freeze_layers_base_model(base_model)

            base_model.classifier = self.classifier_head(1536)
        elif name_pretrained_model == 'efficientnet_b4':
            base_model = models.efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
            base_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

            if self.freeze_layers:
                print("Freeze layers of pretrained model")
                self.freeze_layers_base_model(base_model)

            base_model.classifier = self.classifier_head(1792)
        elif name_pretrained_model == 'efficientnet_b5':
            base_model = models.efficientnet_b5(weights=EfficientNet_B5_Weights.DEFAULT)
            base_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

            if self.freeze_layers:
                print("Freeze layers of pretrained model")
                self.freeze_layers_base_model(base_model)

            base_model.classifier = self.classifier_head(2048)
        elif name_pretrained_model == 'efficientnet_b6':
            base_model = models.efficientnet_b6(weights=EfficientNet_B6_Weights.DEFAULT)
            base_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

            if self.freeze_layers:
                print("Freeze layers of pretrained model")
                self.freeze_layers_base_model(base_model)

            base_model.classifier = self.classifier_head(2304)
        elif name_pretrained_model == 'efficientnet_b7':
            base_model = models.efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT)
            base_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

            if self.freeze_layers:
                print("Freeze layers of pretrained model")
                self.freeze_layers_base_model(base_model)

            base_model.classifier = self.classifier_head(2560)
        elif name_pretrained_model == 'efficientnet_v2_m':
            base_model = models.efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
            base_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

            if self.freeze_layers:
                print("Freeze layers of pretrained model")
                self.freeze_layers_base_model(base_model)

            base_model.classifier = self.classifier_head(1280)
        elif name_pretrained_model == 'efficientnet_v2_s':
            base_model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
            base_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

            if self.freeze_layers:
                print("Freeze layers of pretrained model")
                self.freeze_layers_base_model(base_model)

            base_model.classifier = self.classifier_head(1280)
        elif name_pretrained_model == 'efficientnet_v2_l':
            base_model = models.efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT)
            base_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

            if self.freeze_layers:
                print("Freeze layers of pretrained model")
                self.freeze_layers_base_model(base_model)

            base_model.classifier = self.classifier_head(1280)
        elif name_pretrained_model == 'cct_14_7x2_224':
            base_model = cct_14_7x2_224(pretrained=self.pretrained, progress=self.pretrained, num_classes=self.num_classes, img_size=self.image_size)
            if self.freeze_layers:
                print("Freeze layers of pretrained model")
                self.freeze_layers_base_model(base_model)

        elif name_pretrained_model == 'cct_14_7x2_384':
            base_model = cct_14_7x2_384(pretrained=self.pretrained, progress=self.pretrained, num_classes=self.num_classes, img_size=self.image_size)
            if self.freeze_layers:
                print("Freeze layers of pretrained model")
                self.freeze_layers_base_model(base_model)

        elif name_pretrained_model == 'cct_7_7x2_224_sine':
            base_model = cct_7_7x2_224_sine(pretrained=self.pretrained, progress=self.pretrained, num_classes=self.num_classes, img_size=self.image_size)
            if self.freeze_layers:
                print("Freeze layers of pretrained model")
                self.freeze_layers_base_model(base_model)

        return base_model


def find_last_checkpoint_file(checkpoint_dir, use_best_checkpoint=False):
    '''
    Cerco nella directory checkpoint_dir il file .pth.
    Se use_best_checkpoint = True prendo il best checkpoint
    Se use_best_checkpoint = False prendo quello con l'epoca maggiore tra i checkpoint ordinari
    :param checkpoint_dir:
    :param use_best_checkpoint:
    :return:
    '''
    print("Cerco il file .pth in checkpoint_dir {}: ".format(checkpoint_dir))
    list_file_paths = []

    for file in os.listdir(checkpoint_dir):
        if file.endswith(".pth"):
            path_file = os.path.join(checkpoint_dir, file)
            list_file_paths.append(path_file)
            print("Find: ", path_file)

    print("Number of files .pth: {}".format(int(len(list_file_paths))))
    path_checkpoint = None

    if len(list_file_paths) > 0:

        if use_best_checkpoint:
            if os.path.isfile(os.path.join(checkpoint_dir, 'best.pth')):
                path_checkpoint = os.path.join(checkpoint_dir, 'best.pth')
        else:
            list_epoch_number = []
            for path in list_file_paths:
                file_name = path.split('/')[-1]
                file_name_no_extension = file_name.split('.')[0]

                if file_name_no_extension.split('_')[0] == "checkpoint":
                    number_epoch = int(file_name_no_extension.split('_')[1])
                else:
                    continue
                list_epoch_number.append(number_epoch)
            max_epoch = max(list_epoch_number)
            path_checkpoint = os.path.join(checkpoint_dir, 'checkpoint_' + str(max_epoch) + '.pth')

    return path_checkpoint