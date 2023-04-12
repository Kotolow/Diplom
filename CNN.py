import tensorflow as tf
from datetime import datetime
from matplotlib import pyplot
import itertools
import os
from tqdm import tqdm
from lime import lime_image
import shutil
import random
import math

from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives +
    K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

IMAGE_PATH = "ExfiltrateAttack\\img"
IMAGE_SIZES = (26, 26)
SNAPSHOTS_PATH = f"ExfiltrateAttack/snapshots/snap{IMAGE_SIZES[0]}x{IMAGE_SIZES[1]}"
EPOCH = 5
DATASET_PATH = "ExfiltrateAttack\\dataset"

def remove_dataset():
    print("Cleaning previous dataset...")
    for root, dirs, files in os.walk(DATASET_PATH):
        for file in tqdm(files):
            os.remove(os.path.join(root, file))

def prepare_dataset():
    remove_dataset()
    print(f"Getting new dataset for size {IMAGE_SIZES[0]}x{IMAGE_SIZES[1]}...")
    SPLIT = 0.8
    imagePathsAttack = list()
    imagePathsNormal = list()
    imageFolder = f'{IMAGE_SIZES[0]}x{IMAGE_SIZES[1]}'
    for root, dirs, files in os.walk(IMAGE_PATH):
        for file in files:
            if imageFolder in root:
                if '\\Attack\\' in root:
                    imagePathsAttack.append(os.path.join(root, file))
                else:
                    imagePathsNormal.append(os.path.join(root, file))
    random.shuffle(imagePathsAttack)
    random.shuffle(imagePathsNormal)

    split_index = math.floor(len(imagePathsAttack) * SPLIT)
    training_attack = imagePathsAttack[:split_index]
    testing_attack = imagePathsAttack[split_index:]\

    split_index = math.floor(len(imagePathsNormal) * SPLIT)
    training_normal = imagePathsNormal[:split_index]
    testing_normal = imagePathsNormal[split_index:]

    print(f"Copying training set...")
    training_set = list(itertools.chain(training_normal, training_attack))
    testing_set = list(itertools.chain(testing_normal, testing_attack))

    for imagePath in tqdm(training_set):
        imageName = imagePath.split('\\')[-3:]
        newImagePath = os.path.join(DATASET_PATH, f"train/{imageName[0]}/{imageName[2]}")
        shutil.copy(imagePath, newImagePath)

    print(f"Copying testing set...")
    for imagePath in tqdm(testing_set):
        imageName = imagePath.split('\\')[-3:]
        newImagePath = os.path.join(DATASET_PATH, f"test/{imageName[0]}/{imageName[2]}")
        shutil.copy(imagePath, newImagePath)



if __name__ == "__main__":
    explainer = lime_image.LimeImageExplainer()
    prepare_dataset()
    train_set = tf.keras.utils.image_dataset_from_directory(os.path.join(DATASET_PATH, 'train'), labels='inferred', batch_size=256, image_size=IMAGE_SIZES)
    validation_set = tf.keras.utils.image_dataset_from_directory(os.path.join(DATASET_PATH, 'test'), labels='inferred', batch_size=256, image_size=IMAGE_SIZES)
    #check_set = tf.keras.utils.image_dataset_from_directory(os.path.join(DATASET_PATH, 'check'), labels='inferred', batch_size=1024, image_size=IMAGE_SIZES)
    model = tf.keras.models.Sequential([tf.keras.layers.Rescaling(1./255, input_shape=(IMAGE_SIZES[0], IMAGE_SIZES[1], 3)),
                                        #
                                        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                                        tf.keras.layers.MaxPool2D(2, 2),
                                        #
                                        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                                        tf.keras.layers.MaxPool2D(2, 2),
                                        #
                                        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                                        tf.keras.layers.MaxPool2D(2, 2),
                                        #
                                        tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(128, activation='relu'),
                                        #
                                        tf.keras.layers.Dense(2)
                                        ])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy',f1_m,precision_m, recall_m])
    # model.load_weights(SNAPSHOTS_PATH)
    start = datetime.now()
    history = model.fit(train_set, epochs=EPOCH, validation_data=validation_set)

    end = datetime.now()
    print(f'Training time: {end - start}')
    model.save_weights(SNAPSHOTS_PATH)
    #loss1, acc1 = model.evaluate(check_set)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(EPOCH)

    pyplot.figure(figsize=(8, 8))
    pyplot.subplot(1, 2, 1)
    pyplot.plot(epochs_range, acc, label='Training Accuracy')
    pyplot.plot(epochs_range, val_acc, label='Validation Accuracy')
    pyplot.legend(loc='lower right')
    pyplot.title('Training and Validation Accuracy')

    pyplot.subplot(1, 2, 2)
    pyplot.plot(epochs_range, loss, label='Training Loss')
    pyplot.plot(epochs_range, val_loss, label='Validation Loss')
    pyplot.legend(loc='upper right')
    pyplot.title('Training and Validation Loss')
    pyplot.show()

    #print(f" loss for checking - {loss1}, accuracy for checking {acc1}")