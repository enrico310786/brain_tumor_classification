import argparse
import yaml
import torch
from image_classification_model import ImageClassificationModel
from prettytable import PrettyTable
from torchinfo import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)

def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_config_file', type=str, help='Path of the config file to use')
    opt = parser.parse_args()

    # 2 - load config file
    path_config_file = opt.path_config_file
    print('path_config_file: ', path_config_file)
    cfg = load_config(path_config_file)

    # 3 - load model
    model = ImageClassificationModel(cfg)
    model.to(device)

    print("")
    print(model)
    print("")

    print("Check layers properties")
    for i, properties in enumerate(model.named_parameters()):
        print("Model layer: {} -  name: {} - requires_grad: {} ".format(i, properties[0], properties[1].requires_grad))
    print("")

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("pytorch_total_params: ", pytorch_total_params)
    print("pytorch_total_trainable_params: ", pytorch_total_trainable_params)

    print("")

    count_parameters(model)