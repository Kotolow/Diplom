import pandas
import configparser
from math import sqrt, ceil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import time
import DeepInsight
import Forehead
from sklearn.preprocessing import StandardScaler

def main() -> None:
    config = configparser.ConfigParser()
    config.read('config.txt')

    method = config.get('global', 'method')
    files = config.get('image', 'files')
    files = files.split(',')
    path = config.get('image', 'path')
    try:
        os.mkdir(f'{path}')
    except:
        print('This folder already exist')
    width = int(config.get('image', 'width'))
    height = int(config.get('image', 'height'))

    for file in files:
        if method == 'forehead':
            df = pandas.read_csv(file)
            labels = Forehead.extract_unique_labels(df)
            print(labels)
            startTime = time.time()
            Forehead.to_image(df, path, file)
            print("time:{0:.3f}sec".format(time.time() - startTime))
        elif method == 'deepinsight':
            df = pandas.read_csv(file)
            df_norm = Forehead.min_max_normalization(df)
            unique_labels = Forehead.extract_unique_labels(df_norm)
            print(unique_labels)
            for label in unique_labels:
                try:
                    os.mkdir(f'{path}/{label}')
                except:
                    print('Wrong path or this folder already exist')
            data = []
            labels = []
            for index, row in df_norm.iterrows():
                prepared_row = np.array(np.append(row.values[:-1], [0, 0, 0]), dtype=float)
                data.append(prepared_row)
                labels.append(np.array(row.tail(1))[0])
            data = np.array(data)
            labels = np.array(labels)
            sd = StandardScaler()
            sd.fit(data)
            data = sd.transform(data)
            labels = labels
            data = np.nan_to_num(data)
            deepinsight = DeepInsight.DeepInsight()
            deepinsight.fit(data, method='kpca')
            images = deepinsight.transform(data, width, height)
            #i = 0
            #for image, label in zip(images, labels):
            for i in range(len(labels)):
                plt.imsave(f'{path}/{labels[i]}/{file.split(".")[0]}_{i}.png', images[i], cmap='gray')
                i += 1

if __name__ == "__main__":
    main()