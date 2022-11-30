import random
import shutil
import cv2
import os
import albumentations as A
import pandas as pd

'''
-Per ogni sottodctory suddivido le immagini in train, test e val con percentuali del 60%, 30% e 10%
-Ogni sottodirectory rappresenta una classe a se stante
-Su ciascuna sottodirecotry applico la data augmentation facendo in modo che il numero di immagini in ciascuna sottodirecotry sia lo stesso
-L'aumento delle immagini di train ed eventualmente di test e val è configurabile attraverso il parametro augment
-Lo script è necessario per la generazione dei dataset di train test e val nel formato csv che poi serve per la generazione dei dataloader
'''

###############################################################
# PARAMETERS
###############################################################

#path_delle immagini originali
path_original_dataset = '/mnt/disco_esterno/dataset_trasmissioni_auto/dataset_macchine_v2/original_images/images'

#path in cui inserire le immagini originali suddivise in train test e val
path_dataset_train = '/mnt/disco_esterno/dataset_trasmissioni_auto/dataset_macchine_v2/original_images/immagini_originali_30_classi_ttv/train'
path_dataset_test = '/mnt/disco_esterno/dataset_trasmissioni_auto/dataset_macchine_v2/original_images/immagini_originali_30_classi_ttv/test'
path_dataset_val = '/mnt/disco_esterno/dataset_trasmissioni_auto/dataset_macchine_v2/original_images/immagini_originali_30_classi_ttv/val'

#è la radice del path in cui verranno salvate le immagini aumentate
root = '/home/userpc/Dataset/dataset_macchine_v2/augmented_images_30_classi_v1/'

#path del dataset di train test e val aumentato: root + path_augmented_dataset_train etc etc
path_augmented_dataset_train = 'train'
path_augmented_dataset_test = 'test'
path_augmented_dataset_val = 'val'

#path in cui salvare i dataset csv di train, test e val aumentati: root + path_csv_train_dataset_augmented
path_csv_train_dataset_augmented = os.path.join(root, 'train_dataset_trasmissioni_augmented.csv')
path_csv_test_dataset_augmented = os.path.join(root, 'test_dataset_trasmissioni_augmented.csv')
path_csv_val_dataset_augmented = os.path.join(root, 'val_dataset_trasmissioni_augmented.csv')

PERC_TRAIN = 0.6
PERC_TEST = 0.3

#setto come numero di immagini da raggiungere nelle directory come lacentinaia maggiore più vicina
FINAL_NUMBER_FILE_DIRECTORY_TRAIN = 20 #400
FINAL_NUMBER_FILE_DIRECTORY_TEST = 0
FINAL_NUMBER_FILE_DIRECTORY_VAL = 10 #20

df_train = pd.DataFrame(columns=['BRAND', 'COD_TRASMISSIONE', 'CLASS', 'PATH', 'LABEL'])
df_test = pd.DataFrame(columns=['BRAND', 'COD_TRASMISSIONE', 'CLASS', 'PATH', 'LABEL'])
df_val = pd.DataFrame(columns=['BRAND', 'COD_TRASMISSIONE', 'CLASS', 'PATH', 'LABEL'])

img_size = 224


transform = A.Compose([
    A.CLAHE(p=0.5),
    A.HueSaturationValue(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ToGray(p=0.5),
    A.RandomGamma(p=0.5),
    A.Flip(p=0.5),
    A.Rotate(p=0.5),
    A.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, per_channel=True, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, rotate_limit=45, p=0.5),
    A.Resize(img_size, img_size, p=1.)
])


###############################################################
# FUNZIONI
###############################################################

def make_data_augmentation(path_original_dataset, path_augmented_dataset, final_number, df, class2label,  augment=True):

    #itero su ciscuna direcotry e sottodirectory
    for subdir, dirs, files in os.walk(path_original_dataset):
        for brand in dirs:
            path_brand = os.path.join(path_original_dataset, brand)
            CHECK_FOLDER = os.path.isdir(path_brand)
            if CHECK_FOLDER:
                #print("brand_name: ", brand)
                for sub_subdir, sub_dirs, sub_files in os.walk(path_brand):
                    for cod_trasm in sub_dirs:
                        path_trasm = os.path.join(path_brand, cod_trasm)
                        CHECK_FOLDER = os.path.isdir(path_trasm)
                        if CHECK_FOLDER:
                            #print("cod_trasm: ", cod_trasm)
                            classe = brand + "_" + cod_trasm
                            label = class2label[classe]
                            print("BRAND: {} - COD_TRASMISSIONE: {} - CLASSE: {} - LABEL: {}".format(brand, cod_trasm, classe, label))

                            number_files = len(os.listdir(path_trasm))
                            print("number_files nella directory '{}': {}".format(path_trasm, number_files))

                            path_directory_save = os.path.join(root, path_augmented_dataset, brand, cod_trasm)
                            path_directory_save_for_csv = os.path.join(path_augmented_dataset, brand, cod_trasm)
                            CHECK_FOLDER = os.path.isdir(path_directory_save)
                            if not CHECK_FOLDER:
                                print("Creo la directory '{}'".format(path_directory_save))
                                os.makedirs(path_directory_save)

                            #determino il numero di volte per cui devo applicare la trasformazione su una singola immagine
                            if augment:
                                n_applications = round((final_number-number_files)/number_files)
                                if n_applications < 0:
                                    n_applications = 0
                                print('n_applications: ', n_applications)
                            else:
                                n_applications = 0
                                print('n_applications: ', n_applications)

                            for filename in os.listdir(path_trasm):
                                path_image = os.path.join(path_trasm, filename)
                                image = cv2.imread(path_image)
                                filename = filename.replace(" ", "")
                                filename = "_".join(filename.split('.')[:-1])

                                img = cv2.resize(image, (img_size, img_size))
                                cv2.imwrite(os.path.join(path_directory_save, classe + '_' + str(0) + '_' + filename + '.png'), img)

                                df = df.append({'BRAND': brand,
                                                'COD_TRASMISSIONE': cod_trasm,
                                                'CLASS': classe,
                                                'PATH': os.path.join(path_directory_save_for_csv, classe + '_' + str(0) + '_' + filename + '.png'),
                                                'LABEL': label}, ignore_index=True)

                                for i in range(n_applications):
                                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                    augmented_image = transform(image=image)['image']
                                    augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
                                    cv2.imwrite(os.path.join(path_directory_save, classe + '_' + str(i+1) + '_' + filename + '.png'), augmented_image)
                                    df = df.append({'BRAND': brand,
                                                    'COD_TRASMISSIONE': cod_trasm,
                                                    'CLASS': classe,
                                                    'PATH': os.path.join(path_directory_save_for_csv, classe + '_' + str(i+1) + '_' + filename + '.png'),
                                                    'LABEL': label}, ignore_index=True)

                            number_file_aumentati = len(os.listdir(path_directory_save))
                            print("number_files finali nella directory '{}': {}".format(path_directory_save, number_file_aumentati))
                            print("---------------------------------------------")

    return df


def make_train_test_val_division(path_original_dataset, path_dataset_train, path_dataset_test, path_dataset_val):

    lista_classi = []
    #itero su ciascuna sottodirectory assegnando casualmente il 70% al train, il 20%al test e il 10% al validation
    print("Suddivisione TRAIN - TEST - VAL")

    for subdir, dirs, files in os.walk(path_original_dataset):
        for brand in dirs:
            path_brand = os.path.join(path_original_dataset, brand)
            CHECK_FOLDER = os.path.isdir(path_brand)
            if CHECK_FOLDER:
                #print("brand_name: ", brand)
                for sub_subdir, sub_dirs, sub_files in os.walk(path_brand):
                    for cod_trasm in sub_dirs:
                        path_trasm = os.path.join(path_brand, cod_trasm)
                        CHECK_FOLDER = os.path.isdir(path_trasm)
                        if CHECK_FOLDER:
                            #print("cod_trasm: ", cod_trasm)
                            classe = brand + "_" + cod_trasm
                            #print("classe: ", classe)

                            if classe not in lista_classi:
                                lista_classi.append(classe)

                            print("BRAND: {} - MODELLO TRASMISSIONE: {} - CLASSE: {}".format(brand, cod_trasm, classe))
                            list_filenames_subclass = os.listdir(path_trasm)
                            number_files = len(list_filenames_subclass)
                            index_list = [i for i in range(number_files)]
                            print("number_files nella directory '{}': {}".format(path_trasm, number_files))

                            number_file_train = round(number_files*PERC_TRAIN)
                            number_file_test = round(number_files*PERC_TEST)

                            sum_train_test = number_file_train + number_file_test
                            #se la somma dei file di train e test è uguale al numero totale,
                            #diminuisco di uno il numero dei file di train, così ho spazio per un file di validazione
                            if sum_train_test == number_files:
                                number_file_train = number_file_train - 1

                            train_index_list = random.sample(index_list, k=number_file_train)
                            index_list = list(set(index_list) - set(train_index_list))
                            test_index_list = random.sample(index_list, k=number_file_test)
                            val_index_list = list(set(index_list) - set(test_index_list))

                            # assegnazione TRAIN e copia nella directory train
                            dst_dir = os.path.join(path_dataset_train, brand, cod_trasm)
                            CHECK_FOLDER = os.path.isdir(dst_dir)
                            if not CHECK_FOLDER:
                                os.makedirs(dst_dir)
                            for idx in train_index_list:
                                filename = list_filenames_subclass[idx]
                                src_file = os.path.join(path_trasm, filename)
                                dst_file = os.path.join(dst_dir, filename)
                                #print("Copia immagine da '{}' a {} ".format(src_file, dst_file))
                                shutil.copy2(src_file, dst_file)
                            print("Numero di file nella directory '{}': {}".format(dst_dir, len(os.listdir(dst_dir))))

                            # assegnazione TEST e copia nella directory test
                            dst_dir = os.path.join(path_dataset_test, brand, cod_trasm)
                            CHECK_FOLDER = os.path.isdir(dst_dir)
                            if not CHECK_FOLDER:
                                os.makedirs(dst_dir)
                            for idx in test_index_list:
                                filename = list_filenames_subclass[idx]
                                src_file = os.path.join(path_trasm, filename)
                                dst_file = os.path.join(dst_dir, filename)
                                #print("Copia immagine da '{}' a {} ".format(src_file, dst_file))
                                shutil.copy2(src_file, dst_file)
                            print("Numero di file nella directory '{}': {}".format(dst_dir, len(os.listdir(dst_dir))))

                            # assegnazione VAL e copia nella directory val
                            dst_dir = os.path.join(path_dataset_val, brand, cod_trasm)
                            CHECK_FOLDER = os.path.isdir(dst_dir)
                            if not CHECK_FOLDER:
                                os.makedirs(dst_dir)
                            for idx in val_index_list:
                                filename = list_filenames_subclass[idx]
                                src_file = os.path.join(path_trasm, filename)
                                dst_file = os.path.join(dst_dir, filename)
                                #print("Copia immagine da '{}' a {} ".format(src_file, dst_file))
                                shutil.copy2(src_file, dst_file)
                            print("Numero di file nella directory '{}': {}".format(dst_dir, len(os.listdir(dst_dir))))
                            print('----------------------------------------------------------------')

    return lista_classi


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

###############################################################
# ESECUZIONE
###############################################################

if __name__ == "__main__":
    create_test = True

    # 1 - vedo se esiste la directory di train. Se esiste la cancello e la ricreo. Se non esiste la creo
    clean_create_dir(path_dataset_train)
    # 2 - vedo se esiste la directory di test. Se esiste la cancello e la ricreo. Se non esiste la creo
    clean_create_dir(path_dataset_test)
    # 3 - vedo se esiste la directory di val. Se esiste la cancello e la ricreo. Se non esiste la creo
    clean_create_dir(path_dataset_val)
    # 4 - vedo se esiste la directory di root. Se esiste la cancello e la ricreo. Se non esiste la creo
    clean_create_dir(root)


    #5 - eseguo split train/test/val delle immagini originali e ricavo la lista delle classi
    lista_classi = make_train_test_val_division(path_original_dataset, path_dataset_train, path_dataset_test, path_dataset_val)
    class2label = {k: v for (v, k) in enumerate(lista_classi)}

    print('-----------------------------------------------')
    print('-----------------------------------------------')

    print("lista_classi: ", lista_classi)
    print("")
    print("lista_classi separate da virgola: ", ','.join(lista_classi))
    print("")
    print("class2label: ", class2label)

    print('-----------------------------------------------')
    print('-----------------------------------------------')

    #6 - eseguo data augmentation
    print("Data augmentation")
    print("Train set")
    df_train = make_data_augmentation(path_dataset_train, path_augmented_dataset_train, FINAL_NUMBER_FILE_DIRECTORY_TRAIN, df_train, class2label)
    print("Test set")
    df_test = make_data_augmentation(path_dataset_test, path_augmented_dataset_test, FINAL_NUMBER_FILE_DIRECTORY_TEST, df_test, class2label, augment=False)
    print("Val set")
    df_val = make_data_augmentation(path_dataset_val, path_augmented_dataset_val, FINAL_NUMBER_FILE_DIRECTORY_VAL, df_val, class2label, augment=True)

    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    #7 -  shuffling e salvataggio dei csv
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_test = df_test.sample(frac=1).reset_index(drop=True)
    df_val = df_val.sample(frac=1).reset_index(drop=True)

    df_train.to_csv(path_csv_train_dataset_augmented, index=False)
    df_test.to_csv(path_csv_test_dataset_augmented, index=False)
    df_val.to_csv(path_csv_val_dataset_augmented, index=False)

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
    print("")
    print(df_train[["CLASS"]].value_counts())
    print('-------------------------------------------------------------')
    print("")
    print(df_test[["CLASS"]].value_counts())
    print('-------------------------------------------------------------')
    print("")
    print(df_val[["CLASS"]].value_counts())