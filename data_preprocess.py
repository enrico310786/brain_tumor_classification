import random
import shutil
import os
from distutils.dir_util import copy_tree
import pandas as pd


###############################################################
# PARAMETERS
###############################################################

#path_of original dataset
path_original_dataset_train = "/mnt/disco_esterno/brain_tumor_dataset/brain_tumor_images/Training"
path_original_dataset_test = "/mnt/disco_esterno/brain_tumor_dataset/brain_tumor_images/Testing"

#path of final dataset
root = "/mnt/disco_esterno/brain_tumor_dataset/brain_tumor_images/Final_Dataset_V1"
path_dataset_train = os.path.join(root, 'train')
path_dataset_test = os.path.join(root, 'test')
path_dataset_val = os.path.join(root, 'val')

# path of csv files
path_csv_train = os.path.join(root, 'train.csv')
path_csv_test = os.path.join(root, 'test.csv')
path_csv_val = os.path.join(root, 'val.csv')

PERC_TRAIN = 0.8

df_train = pd.DataFrame(columns=['CLASS', 'PATH', 'LABEL'])
df_test = pd.DataFrame(columns=['CLASS', 'PATH', 'LABEL'])
df_val = pd.DataFrame(columns=['CLASS', 'PATH', 'LABEL'])


###############################################################
# FUNZIONI
###############################################################

def create_csv(path_dataset, df, class2label, type_dataset):
    for subdir, dirs, files in os.walk(path_dataset):
        for cl in dirs:
            path_cl = os.path.join(path_dataset, cl)
            CHECK_FOLDER = os.path.isdir(path_cl)
            if CHECK_FOLDER:
                print("class_name: ", cl)
                label = class2label[cl]
                list_filenames_food = os.listdir(path_cl)
                for file_name in list_filenames_food:
                    relative_path = os.path.join(type_dataset, cl, file_name)
                    df = df.append({'CLASS': cl,
                                    'PATH': relative_path,
                                    'LABEL': label}, ignore_index=True)

    return df


def make_train_val_division(path_original_dataset, path_dataset_train, path_dataset_val):

    list_classes = []
    print(" TRAIN/ VAL SPLIT")

    for subdir, dirs, files in os.walk(path_original_dataset):
        for cl in dirs:
            path_cl = os.path.join(path_original_dataset, cl)
            CHECK_FOLDER = os.path.isdir(path_cl)
            if CHECK_FOLDER:
                print("class_name: ", cl)
                list_classes.append(cl)
                list_filenames_cl = os.listdir(path_cl)
                number_files = len(list_filenames_cl)
                index_list = [i for i in range(number_files)]
                print("number_files in directory '{}': {}".format(path_cl, number_files))

                number_file_train = round(number_files * PERC_TRAIN)
                train_index_list = random.sample(index_list, k=number_file_train)
                val_index_list = list(set(index_list) - set(train_index_list))

                # copy files in train directory
                dst_dir = os.path.join(path_dataset_train, cl)
                CHECK_FOLDER = os.path.isdir(dst_dir)
                if not CHECK_FOLDER:
                    os.makedirs(dst_dir)
                for idx in train_index_list:
                    filename = list_filenames_cl[idx]
                    src_file = os.path.join(path_cl, filename)
                    dst_file = os.path.join(dst_dir, filename)
                    shutil.copy2(src_file, dst_file)
                print("Number of files in dir '{}': {}".format(dst_dir, len(os.listdir(dst_dir))))

                # copy files in val directory
                dst_dir = os.path.join(path_dataset_val, cl)
                CHECK_FOLDER = os.path.isdir(dst_dir)
                if not CHECK_FOLDER:
                    os.makedirs(dst_dir)
                for idx in val_index_list:
                    filename = list_filenames_cl[idx]
                    src_file = os.path.join(path_cl, filename)
                    dst_file = os.path.join(dst_dir, filename)
                    shutil.copy2(src_file, dst_file)
                print("Number of files in dir '{}': {}".format(dst_dir, len(os.listdir(dst_dir))))
                print('----------------------------------------------------------------')

    return list_classes


def clean_create_dir(path_dir):

    CHECK_FOLDER = os.path.isdir(path_dir)
    if CHECK_FOLDER:
        print("The directory '{}' exists. Delete it".format(path_dir))
        try:
            shutil.rmtree(path_dir)
        except OSError as e:
            print("Error: {}".format(e.strerror))
            raise e

        CHECK_FOLDER = os.path.isdir(path_dir)
        if not CHECK_FOLDER:
            print("Create the directory '{}'".format(path_dir))
            os.makedirs(path_dir)
    else:
        print("Create the directory '{}'".format(path_dir))
        os.makedirs(path_dir)


###############################################################
# EXECUTION
###############################################################


if __name__ == "__main__":

    # 1 - clean and create the dataset directory
    clean_create_dir(root)

    # 2 - copy in the dataset directory the "Testing" dataset renaiming it as Test
    copy_tree(path_original_dataset_test, path_dataset_test)

    # 3 - do split train/val of the train dataset
    list_classes = make_train_val_division(path_original_dataset_train, path_dataset_train, path_dataset_val)
    class2label = {k: v for (v, k) in enumerate(list_classes)}

    print('-----------------------------------------------')
    print('-----------------------------------------------')

    print("list_classes: ", list_classes)
    print("")
    print("list_classes: ", ','.join(list_classes))
    print("")
    print("class2label: ", class2label)

    print('-----------------------------------------------')
    print('-----------------------------------------------')

    # 4 - create csv files
    print("Create csv files")
    print("Train set")
    df_train = create_csv(path_dataset_train, df_train, class2label, "train")
    print("Test set")
    df_test = create_csv(path_dataset_test, df_test, class2label, "test")
    print("Val set")
    df_val = create_csv(path_dataset_val, df_val, class2label, "val")

    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    # 5 -  shuffling and saving of csv
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_test = df_test.sample(frac=1).reset_index(drop=True)
    df_val = df_val.sample(frac=1).reset_index(drop=True)

    df_train.to_csv(path_csv_train, index=False)
    df_test.to_csv(path_csv_test, index=False)
    df_val.to_csv(path_csv_val, index=False)

    print("df_train info")
    print(df_train.info())
    print('-------------------------------------------------------------')
    print("")
    print("df_test info")
    print(df_test.info())
    print('-------------------------------------------------------------')
    print("")
    print("df_val info")
    print(df_val.info())

    print('-------------------------------------------------------------')
    print("df_train:  CLASS values count")
    print(df_train[["CLASS"]].value_counts())
    print('-------------------------------------------------------------')
    print("df_test:  CLASS values count")
    print(df_test[["CLASS"]].value_counts())
    print('-------------------------------------------------------------')
    print("df_val:  CLASS values count")
    print(df_val[["CLASS"]].value_counts())