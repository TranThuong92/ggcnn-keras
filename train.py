from __future__ import print_function
import numpy as np
import keras.backend as K
import tensorflow as tf
import os
import glob
import matplotlib.pyplot as plt

from models.ggcnn_keras import ggcnn
from models.ggcnn_att import ggcnn_att
from models.ggcnn2 import ggcnn2
from models.ggcnn_att2 import ggcnn_att2
from utils.data.cornelldataset import cornell_data
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import argparse

file_path = 'Dataset' #Need to convert to argparse later

def parse_args():
    parser = argparse.ArgumentParser(description='Train GG-CNN')

    # Network
    parser.add_argument('--network', type=str, default='ggcnn', help='Network Name in .models')
    parser.add_argument('--pretrained', type=str, default=False, help='Load pretrained model or not!')

    args = parser.parse_args()
    return args

def run():
    args = parse_args()

    # Prepare data
    dataset = cornell_data(file_path)
    split = 0.9
    traindata = cornell_data(file_path, end=0.9)
    valdata = cornell_data(file_path, start=0.9)

    if args.network == "ggcnn2":
        # Model
        ggcnn_model = ggcnn2()
        model = ggcnn_model.model()
        loss = ggcnn_model.compute_loss
        if args.pretrained:
            model.load_weights('ggcnn_weight_2.h5')
    elif args.network == "ggcnn":
        # Model
        ggcnn_model = ggcnn()
        model = ggcnn_model.model()
        loss = ggcnn_model.compute_loss
        if args.pretrained:
            model.load_weights('ggcnn_weight.h5')
    elif args.network == "ggcnn_att":
        # Model
        ggcnn_model = ggcnn_att()
        model = ggcnn_model.model()
        loss = ggcnn_model.compute_loss
        if args.pretrained:
            model.load_weights('ggcnn_weight_att.h5')
    elif args.network == "ggcnn_att2":
        # Model
        ggcnn_model = ggcnn_att()
        model = ggcnn_model.model()
        loss = ggcnn_model.compute_loss
        if args.pretrained:
            model.load_weights('ggcnn_weight_att2.h5')
    else:
        ValueError("The network must be 'ggcnn', 'ggcnn2' or 'ggcnn_att'!")

    # Optimizers
    #opt = optimizers.Adam( lr = 1e-5,
    #                   beta_1 = 0.9,
    #                   beta_2 = 0.999,
    #                   amsgrad = False
    #                 )
    #opt = optimizers.Adam()
    model.compile(optimizer = 'adadelta', loss=loss)

    tb_counter  = len([log for log in os.listdir(os.path.expanduser('~/logs_obt/')) if 'obt_' in log]) + 1
    tensorboard = TensorBoard(log_dir=os.path.expanduser('~/logs_obt/') + 'obt_' + '_' + str(tb_counter),
                          histogram_freq=0,
                          write_graph=True,
                          write_images=False)

    if args.network == 'ggcnn':
        checkpoint = ModelCheckpoint('ggcnn_weight.h5',
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min',
                                 period=1)
    elif args.network == 'ggcnn2':
        checkpoint = ModelCheckpoint('ggcnn_weight_2.h5',
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min',
                                 period=1)
    elif args.network == 'ggcnn_att':
        checkpoint = ModelCheckpoint('ggcnn_weight_att.h5',
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min',
                                 period=1)
    elif args.network == 'ggcnn_att2':
        checkpoint = ModelCheckpoint('ggcnn_weight_att2.h5',
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min',
                                 period=1)

    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

    model.fit_generator(generator       = traindata,
                    steps_per_epoch  = len(traindata),
                    epochs           = 30,
                    verbose          = 1,
                    validation_data  = valdata,
                    validation_steps = len(valdata),
                    callbacks        = [early_stop, checkpoint, tensorboard],
                    #callbacks        = [checkpoint, tensorboard],
                    max_queue_size   = 5)

if __name__ == "__main__":
    run()
