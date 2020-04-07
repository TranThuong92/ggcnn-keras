from __future__ import print_function

from models.ggcnn_keras import ggcnn
from models.ggcnn2 import ggcnn2
from models.ggcnn_att import ggcnn_att
import numpy as np
import keras.backend as K
import tensorflow as tf
import os
import glob
import matplotlib.pyplot as plt
import keras
import argparse

from utils.data.cornelldataset import cornell_data
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from utils.data_processing.postprocessing import process_output


file_path = 'Dataset' #Need to convert to argparse later

def parse_args():
    parser = argparse.ArgumentParser(description='Train GG-CNN')

    # Network
    parser.add_argument('--network', type=str, default='ggcnn', help='Network Name in .models')

    args = parser.parse_args()
    return args

def evaluate():
    args = parse_args()
    # Prepare data
    dataset = cornell_data(file_path)
    split = 0.9
    traindata = cornell_data(file_path, end=0.9)
    valdata = cornell_data(file_path, start=0.9)
    # Load model
    if args.network == "ggcnn2":
        # Model
        ggcnn_model = ggcnn2()
        model = ggcnn_model.model()
        loss = ggcnn_model.compute_loss
        model.load_weights('ggcnn_weight_2.h5')
    elif args.network == "ggcnn":
        # Model
        ggcnn_model = ggcnn()
        model = ggcnn_model.model()
        loss = ggcnn_model.compute_loss
        model.load_weights('ggcnn_weight.h5')
    elif args.network == "ggcnn_att":
        # Model
        ggcnn_model = ggcnn_att()
        model = ggcnn_model.model()
        loss = ggcnn_model.compute_loss
        model.load_weights('ggcnn_weight_att.h5')
    elif args.network == "ggcnn_att2":
        # Model
        ggcnn_model = ggcnn_att()
        model = ggcnn_model.model()
        loss = ggcnn_model.compute_loss
        model.load_weights('ggcnn_weight_att2.h5')
    else:
        ValueError("The network must be 'ggcnn' or 'ggcnn_att'!")
    # True
    for i in range(0, len(valdata)):
        rgb_img, output = valdata[i]
        pos_img = output[...,0]
        width_img = output[...,1]
        cos_img = output[...,2]
        sin_img = output[...,3]
        # Pred
        pred  = model.predict(rgb_img)
        pred_pos = pred[...,0]
        pred_width = pred[...,1]
        pred_cos = pred[...,2]
        pred_sin = pred[...,3]

        fig = plt.figure(figsize=(20, 5))
        ax = fig.add_subplot(1, 8, 1)
        ax.imshow(rgb_img[0])
        ax.axis('off')
        plt.title('rgb')

        ax = fig.add_subplot(1, 8, 2)
        ax.imshow(pos_img[0], cmap='jet')
        ax.axis('off')
        plt.title('pos_true')

        ax = fig.add_subplot(1, 8, 3)
        ax.imshow(pred_pos[0], cmap='jet', vmin=0, vmax=1)
        ax.axis('off')
        plt.title('pos_pred')

        ax = fig.add_subplot(1, 8, 4)
        ax.imshow(pred_width[0], cmap='jet', vmin=0, vmax=1)
        ax.axis('off')
        plt.title('width_pred')

        ax = fig.add_subplot(1, 8, 5)
        ax.imshow(cos_img[0], cmap='jet', vmin=-1, vmax=1)
        ax.axis('off')
        plt.title('cos_true')

        ax = fig.add_subplot(1, 8, 6)
        ax.imshow(pred_cos[0], cmap='jet', vmin=-1, vmax=1)
        ax.axis('off')
        plt.title('cos_pred')

        ax = fig.add_subplot(1, 8, 7)
        ax.imshow(sin_img[0], cmap='jet', vmin=-1, vmax=1)
        ax.axis('off')
        plt.title('sin_true')

        ax = fig.add_subplot(1, 8, 8)
        ax.imshow(pred_sin[0], cmap='jet', vmin=-1, vmax=1)
        ax.axis('off')
        plt.title('sin_pred')

        display_img_true = process_output(rgb_img[0], pred_pos[0], cos_img[0], sin_img[0], width_img[0])
        display_img_pred = process_output(rgb_img[0], pred_pos[0], pred_cos[0], pred_sin[0], pred_width[0])

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(display_img_true)
        ax.axis('off')
        ax = fig.add_subplot(1, 2, 2)
        ax.imshow(display_img_pred)
        ax.axis('off')

        plt.show()

if __name__ == "__main__":
    evaluate()
