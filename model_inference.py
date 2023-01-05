import pandas as pd
import yaml
import torch
import cv2
from torchvision import transforms


##############################################
# PARAMETERS AND FUNCTIONS
##############################################
from image_classification_model import ImageClassificationModel

image_test_path = "/mnt/disco_esterno/brain_tumor_dataset/brain_tumor_images/Final_Dataset_V1/test/meningioma_tumor/image(2).jpg"
train_dataset_csv = "dataset/Final_Dataset_V1/train.csv"
path_best_checkpoint = "exps_classifier_brain_tumor/efficientnet_b4/models/best.pth"
path_config_file = "config/classifier_efficientnet_b4.yaml"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
post_act = torch.nn.Softmax(dim=1)


def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


class2label = {}
# go through the lines of the dataset
df_dataset_train = pd.read_csv(train_dataset_csv)
for index, row in df_dataset_train.iterrows():
    class_name = row["CLASS"]
    label = row["LABEL"]

    if class_name not in class2label:
        class2label[class_name] = label

# sort the value of the label
class2label = dict(sorted(class2label.items(), key=lambda item: item[1]))
print("class2label: ", class2label)
print("")
label2class = {k: v for (v, k) in class2label.items()}
print("label2class: ", label2class)


##############################
# LOAD MODEL
###############################

print('path_config_file: ', path_config_file)
cfg = load_config(path_config_file)

print("Load the model")
model = ImageClassificationModel(cfg)

print("Load the best checkpoint: ", path_best_checkpoint)
checkpoint = torch.load(path_best_checkpoint, map_location=torch.device(device))
model.load_state_dict(checkpoint['model'])
model = model.eval()
model = model.to(device)

##################################
# INFERENCE
##################################

# load the image_byte
image = cv2.imread(image_test_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (cfg["data"]["size"], cfg["data"]["size"]), interpolation=cv2.INTER_AREA)
image = transforms.ToTensor()(image)
image = normalize(image)
image = image[None].to(device)

with torch.no_grad():
    outputs = model(image)
    # Get the predicted label
    preds = post_act(outputs)
    pred_values, pred_labels = torch.max(preds, 1)
    predicted_score = round(pred_values.cpu().numpy().tolist()[0], 4)
    predicted_class = label2class[pred_labels.cpu().numpy().tolist()[0]]
    print("predicted_class: ", predicted_class)
    print("predicted_score: ", predicted_score)


