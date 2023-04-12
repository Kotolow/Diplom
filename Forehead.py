import pandas
from math import sqrt, ceil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import time
import configparser


def load_df(file):
    df = pandas.read_csv(file)
    return df

def extract_unique_labels(df):
    unique_labels = []
    for i in df[' Label']:
        if i not in unique_labels:
            unique_labels.append(i)
    return unique_labels

def min_max_normalization(df):
    result = df.copy()
    for feature_name in df.columns:
        if feature_name==' Label':
            continue
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        if min_value == max_value:
            result[feature_name] = df[feature_name]
        else:
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

def to_image(df, path, file, blw=True):
    df_norm = min_max_normalization(df)
    unique_labels = extract_unique_labels(df_norm)
    for label in unique_labels:
        try:
            os.mkdir(f'{path}/{label}')
        except:
            print('Wrong path or this folder already exist')

    for index, row in df_norm.iterrows():
        label_path = np.array(row.tail(1))[0]
        side_len = ceil(sqrt(len(row)))
        prepared_row = np.array(np.append(row.values[:-1], [0,0,0]), dtype=float)
        if blw:
            image = Image.fromarray(np.uint8(prepared_row.reshape(side_len, side_len)*255)).resize((side_len, side_len))
            image.save(f'{path}/{label_path}/{file.split(".")[0]}_{index}.png')
        else:
            fig=plt.figure(figsize=(side_len,side_len), dpi=50)
            plt.axis('off')
            plt.imsave(f'{path}/{label_path}/{file.split(".")[0]}_{index}.png', prepared_row.reshape(side_len,side_len))