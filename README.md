# Brain Tumor Classification (MRI)

In this project we consider images of the dataset hosted by Kaggle **Brain Tumor Classification (MRI)**.
You can read more about the dataset and download it visiting the Kaggle page
at this [link](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri?select=Training).

The dataset contains brain images acquired by Magnetic Resonance distributed in four classes: 
glioma_tumor, meningioma_tumor, no_tumor and pituitary_tumor and 
is well suited to the purpose of performing an image classification task.

Thanks to this project you can use a series of pretrained model between 
ResNet, EfficientNet, EfficientNet_V2 and Compact Convolutional Transformer architectures
that you can fine tune on this custom dataset.

You can read more about this project on the following Medium post:

## Preproces

When you download and unzip the dataset, you can see that is already splitted in a training and testing directories.
Each directory contains four sub-directory, one for each class of images

The dataset preprocess consists of the following steps:

1) split the trin set in a train and validation subsets
2) assign each class a different label from 0 to 3
3) create the csv files for train, validation and test dataset with the path to the images and the correct label.
   These files will be used to create the dataloaders.

The preprocess is runned by the script

```bash
python data_preprocess.py
```
Note that during the execution will be generated the dictionary class2label, namely the map from the name of the
category to the corresponding label. The order of the keys of the dictionary, that are the classes of the dataset, has to be respected
at the inference time to obtain the correct classes.

## Train images augmentation

Once splitted the train, val and test dataset, you can augment the images of the train dataset to generate new synthetic images.
Running the script

```bash
python augment_train_dataset.py
```

you perform image data augmentation by Albumentations. You can set the final number of the images for each category.
The script will generate a new train folder with the original and the new synthetic images for each class. 
Moreover, the script will generate the .csv file corresponding to the new augmented train dataset.

## Train and test

If you want to train and test the classifier model on your device, you can run the training and test script
```bash
python run_train_test.py
```
This only requires the configuration file containing the values of the hyperparameters 
of the model and the paths to the datasets.

If you want to make the train procedure on AWS Batch platform, you can use the script
```bash
python run_train_test_aws.py
```
In addition to the configuration file path, you must also enter the name of the S3 bucket 
and the bucket directory where are contained the datasets. 
The download and upload of the data and checkpoints saved during the training is managed 
by the boto3 client using the multidown and multiup functions.
In this case you need to create the docker image of the project, inserting inside also the
credential file with your aws_access_key_id and aws_secret_access_key.

To create the docker image you can use the provided Dockerfile

In the configuration file you can set

1) the path to the dataset and the csv files for train, validation and test. Note that these files
   have to stay inside the dataset directory
2) batch_size: the size of the batch used by the dataloaders. 
   Depending on the RAM memory of your device, if this value is too high you can obtain CUDA out of memory errors
3) name_pretrained_model: the name of the pretrained model to use. You can use
   
   a) efficientnet_b0 to efficientnet_b7 
   
   b) resnet18, resnet34, resnet50, resnet101, resnet152

   c) efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l

   d) cct_14_7x2_224, cct_14_7x2_384

4) saving_dir_experiments: the path where are saved the checkpoints and the results
5) saving_dir_model: the subdir where are saved the checkpoints and the results 
6) num_classes: the number of classes to be classified
7) num_epoch: the number of epochs of the training phase
8) learning_rate: the initial learning rate
9) scheduler_step_size: the number of epochs that must pass to change the value of the learning rate
10) scheduler_gamma: the value by which the current learning rate is multiplied to obtain the next learning rate 
11) freeze_layers: 1 if the weights of the base model have to be freezed. 0 otherwise
12) epoch_start_unfreeze: the epoch from which you want to unlock the base model weights
13) layer_start_unfreeze: the base model layer from which you intend to unlock the weights
14)  n_nodes: the number of nodes of the hidden layer between the base model and the output layer
15) do_train: 1 if you want to train the model. 0 otherwise
16) do_test: 1 if you want to test the model. 0 otherwise
17) size: the size of the image
18) do_resize: 1 if apply a resize using the chosen size. 0 otherwise
19) normalization: "imagenet" to apply the imagenet normalization; "pm1" to normalize the image with pixel values in the range [-1, 1]; "None" no normalization will be applied

The size of the image depends on the chosen model. You can follow this table to choose the correct size

Model  | Image Size
------------- | -------------
 All ResNet  | 224x224
 EfficientNet-B0 | 224x224
 EfficientNet-B1 |  240x240
 EfficientNet-B2 |  288x288
 EfficientNet-B3 |  300x300
 EfficientNet-B4 |  380x380
 EfficientNet-B5 |  456x456
 EfficientNet-B6 |  528x528
 EfficientNet-B7 |  600x600
 EfficientNet-V2-s|  384x384
 EfficientNet-V2-m |  480x480
 CCT-14_7x2_224 |  224x224
 CCT-14_7x2_384 |  384x384

## Inference

One trained your model, with the script 
```bash
python model_inference.py
```
you can apply inference to some images and check the result

## Trained models

You can download my trained models at the following links

[EfficientNet-B4](https://drive.google.com/file/d/103Z_c028sqbxGVeAUCUuH2jscgRcp9Cl/view?usp=share_link)

[Resnet50](https://drive.google.com/file/d/1-4iJ_Ntk4_LBtWMDM4OO97_YzJAigPyZ/view?usp=sharing)

## Environment

I use Python 3.7.9

To run the scripts in your device create an environment using the file 'requirements.txt'

To run the script on AWS use the file 'requirements_docker.txt' as expressed in the Dockerfile


