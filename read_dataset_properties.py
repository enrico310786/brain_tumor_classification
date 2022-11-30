"""
Lo script analizza il contenuto del dataset originale.
La struttura deve essere del tipo

images:
    brand_1:
        cod_trasm_1_brand_1:
            img_1
            img_2
            img_3
            ...
        cod_trasm_2_brand_1:
            img_1
            img_2
            img_3
            ...
        ...
    brand_2:
        cod_trasm_1_brand_2:
            img_1
            img_2
            img_3
            ...
        cod_trasm_2_brand_2:
            img_1
            img_2
            img_3
            ...
        ...
    brand_3:
        cod_trasm_1_brand_3:
            img_1
            img_2
            img_3
            ...
        cod_trasm_2_brand_3:
            img_1
            img_2
            img_3
            ...
        ...
"""


import pandas as pd
import os


def read_dataset(path_dataset):

    #creo un dataframe vuoto
    df = pd.DataFrame(columns=['BRAND', 'COD_TRASMISSIONE', 'CLASSE', 'NUM_IMAGES'])
    df['NUM_IMAGES'] = df['NUM_IMAGES'].astype(int)
    num_tot_images = 0

    for subdir, dirs, files in os.walk(path_dataset):
        for brand_name in dirs:
            path_subdir = os.path.join(path_dataset, brand_name)
            CHECK_FOLDER = os.path.isdir(path_subdir)
            if CHECK_FOLDER:
                print("brand_name: ", brand_name)
                for sub_subdir, sub_dirs, sub_files in os.walk(path_subdir):
                    for cod_trasm in sub_dirs:
                        path_sub_dirs = os.path.join(path_subdir, cod_trasm)
                        CHECK_FOLDER = os.path.isdir(path_sub_dirs)
                        if CHECK_FOLDER:
                            print("cod_trasm: ", cod_trasm)
                            classe = brand_name + "_" + cod_trasm
                            print("classe: ", classe)
                            number_files = len(os.listdir(path_sub_dirs))
                            print("number_files nella directory '{}': {}".format(path_sub_dirs, number_files))

                            num_tot_images += number_files
                            df = df.append({'BRAND': brand_name,
                                            'COD_TRASMISSIONE': cod_trasm,
                                            'CLASSE': classe,
                                            'NUM_IMAGES': number_files}, ignore_index=True)

                print('--------------------')

    return df, num_tot_images

if __name__ == '__main__':

    path_dataset_info = 'dataset/dataset_original_30_classi.csv'
    path_dataset = '/mnt/disco_esterno/dataset_trasmissioni_auto/dataset_macchine_v2/original_images/images'

    df, num_tot_images = read_dataset(path_dataset)
    df.to_csv(path_dataset_info, index=False)

    print(df.info())
    print('--------------------')

    print('Numero totale di immagini: ')
    print(num_tot_images)
    print('--------------------')

    print("Brand univoci: ")
    print(list(df['BRAND'].unique()))
    print('--------------------')

    print("Numero di brand univoci: ")
    print(len(list(df['BRAND'].unique())))
    print('--------------------')

    print("Numero di trasmissioni per brand")
    print(df.groupby(["BRAND"]).size())
    print("--------------------")

    print("Classi univoche: ")
    print(list(df['CLASSE'].unique()))
    print('--------------------')

    print("Numero di classi univoche: ")
    print(len(list(df['CLASSE'].unique())))
    print('--------------------')

    print("Classe con numero maggiore di immagini: ")
    print(df.iloc[df['NUM_IMAGES'].idxmax()])
    print('--------------------')

    print("Classe con numero minore di immagini: ")
    print(df.iloc[df['NUM_IMAGES'].idxmin()])

