import os
import shutil
import pandas as pd
import cv2
import albumentations as A

transform = A.Compose([
    A.HueSaturationValue(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.RandomGamma(p=0.5),
    A.Flip(p=0.5),
    A.Rotate(p=0.5),
    A.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, per_channel=True, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, rotate_limit=45, p=0.5),
    A.Transpose(p=0.5),
])


def clean_create_dir(path_dir):

    CHECK_FOLDER = os.path.isdir(path_dir)
    if CHECK_FOLDER:
        print("La directory '{}' esiste. La cancello".format(path_dir))
        try:
            shutil.rmtree(path_dir)
        except OSError as e:
            print("Error: {}".format(e.strerror))
            raise e

        CHECK_FOLDER = os.path.isdir(path_dir)
        if not CHECK_FOLDER:
            print("Creo la directory '{}'".format(path_dir))
            os.makedirs(path_dir)
    else:
        print("Creo la directory '{}'".format(path_dir))
        os.makedirs(path_dir)


def make_data_augmentation(path_original_dataset, path_augmented_dataset, root_csv, final_number, df, class2label):

    #iter over directory
    for subdir, dirs, files in os.walk(path_original_dataset):
        for classe in dirs:
            path_class = os.path.join(path_original_dataset, classe)
            CHECK_FOLDER = os.path.isdir(path_class)
            if CHECK_FOLDER:
                label = class2label[classe]
                print("CLASS: {}  - LABEL: {}".format(classe, label))
                number_files = len(os.listdir(path_class))
                print("number_files nella directory '{}': {}".format(path_class, number_files))

                path_directory_save = os.path.join(path_augmented_dataset, classe)
                path_directory_save_for_csv = os.path.join(root_csv, classe)
                CHECK_FOLDER = os.path.isdir(path_directory_save)
                if not CHECK_FOLDER:
                    print("Create dir directory '{}'".format(path_directory_save))
                    os.makedirs(path_directory_save)

                #determino il numero di volte per cui devo applicare la trasformazione su una singola immagine

                n_applications = round((final_number-number_files)/number_files)
                if n_applications < 0:
                    n_applications = 0
                print('n_applications: ', n_applications)

                for filename in os.listdir(path_class):
                    path_image = os.path.join(path_class, filename)
                    image = cv2.imread(path_image)
                    filename_no_ext, extension= filename.split(".")[0], filename.split(".")[-1]

                    # copy the original image from the sourse dir to the dest dir
                    dst_file = os.path.join(path_directory_save, filename)
                    shutil.copy2(path_image, dst_file)

                    df = df.append({'CLASS': classe,
                                    'PATH': os.path.join(path_directory_save_for_csv, filename),
                                    'LABEL': label}, ignore_index=True)

                    for i in range(n_applications):
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        augmented_image = transform(image=image)['image']
                        augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
                        new_file_name = filename_no_ext + '_' + str(i+1) + '.' + extension
                        dst_file = os.path.join(path_directory_save, new_file_name)

                        cv2.imwrite(dst_file, augmented_image)
                        df = df.append({'CLASS': classe,
                                        'PATH': os.path.join(path_directory_save_for_csv, new_file_name),
                                        'LABEL': label}, ignore_index=True)

                number_file_aumentati = len(os.listdir(path_directory_save))
                print("Final number of files on dir '{}': {}".format(path_directory_save, number_file_aumentati))
                print("---------------------------------------------")

    return df


if __name__ == "__main__":

    FINAL_NUMBER_TRAIN_CLASSES = 5000
    path_dataset_train = "/mnt/disco_esterno/brain_tumor_dataset/brain_tumor_images/Final_Dataset_V1/train"
    path_augmented_dataset_train = "/mnt/disco_esterno/brain_tumor_dataset/brain_tumor_images/Final_Dataset_V1/train_augmented"
    path_augmented_csv_train = "/mnt/disco_esterno/brain_tumor_dataset/brain_tumor_images/Final_Dataset_V1/train_augmented.csv"
    df_train_augmented = pd.DataFrame(columns=['CLASS', 'PATH', 'LABEL'])
    root_csv = "train_augmented"

    # the list of classes has to be on the same order as the original dataset
    list_classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
    class2label = {k: v for (v, k) in enumerate(list_classes)}

    # 1 - clean and create the dataset directory
    clean_create_dir(path_augmented_dataset_train)

    # 2 - make data augmentation
    print("Data augmentation")
    print("Train set")
    df_train = make_data_augmentation(path_dataset_train, path_augmented_dataset_train, root_csv, FINAL_NUMBER_TRAIN_CLASSES, df_train_augmented, class2label)

    # 3 - create new csv
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_train.to_csv(path_augmented_csv_train, index=False)
    print("df_train info")
    print(df_train.info())
    print('-------------------------------------------------------------')
    print("df_train:  CLASS values count")
    print(df_train[["CLASS"]].value_counts())
    print('-------------------------------------------------------------')