from __future__ import print_function
import numpy as np
import keras.backend as K
import cv2
import tensorflow as tf
import os
import glob
import matplotlib.pyplot as plt

from keras.utils import Sequence
from skimage.io import imread
from skimage.draw import polygon
from .axiliary import DepthImage, RgbImage, GraspRectangles

class GraspDatasetBase(Sequence):
    """
    Abstract grasp database for GGCNN
    """
    def __init__(self, output_size=300, include_depth=True, include_rgb=False, random_rotate=False,
                 random_zoom=False, input_only=False):
        """
        :param output_size: Image output size in pixels (square)
        :param include_depth: Whether depth image is included
        :param include_rgb: Whether RGB image is included
        :param random_rotate: Whether random rotations are applied
        :param random_zoom: Whether random zooms are applied
        :param input_only: Whether to return only the network input (no labels)
        """
        self.output_size = output_size
        self.random_rotate = random_rotate
        self.random_zoom = random_zoom
        self.input_only = input_only
        self.include_depth = include_depth
        self.include_rgb = include_rgb

        if include_depth is False and include_rgb is False:
            raise ValueError('At least one of Depth or RGB must be specified.')

    def __len__(self):
        return len(self.grasp_files)

    def get_depth(self, idx):
        raise NotImplementedError()

    def get_rgb(self, idx):
        raise NotImplementedError()

    def get_gtbb(self, idx):
        raise NotImplementedError()

    def __getitem__(self, idx):
        # Load the depth image
        depth_img = self.get_depth(idx)

        # Load the rgb image
        rgb_img = self.get_rgb(idx)

        # Load the grasps
        grs = self.get_gtbb(idx)
        pos_out, width_out, cos_out, sin_out = grs.draw((self.output_size, self.output_size))

        # Expand dim
        depth_img = np.expand_dims(depth_img, axis=0)
        depth_img = np.expand_dims(depth_img, axis=3)
        rgb_img = np.expand_dims(rgb_img, axis=0)
        pos_out = np.expand_dims(pos_out, axis=0)
        pos_out = np.expand_dims(pos_out, axis=3)
        width_out = np.expand_dims(width_out, axis=0)
        width_out = np.expand_dims(width_out, axis=3)
        cos_out = np.expand_dims(cos_out, axis=0)
        cos_out = np.expand_dims(cos_out, axis=3)
        sin_out = np.expand_dims(sin_out, axis=0)
        sin_out = np.expand_dims(sin_out, axis=3)
        #print(sin_out.shape)
        #print(cos_out.shape)

        output = np.concatenate((pos_out, width_out, cos_out, sin_out), axis=3)
        #return depth_img, rgb_img, grs, pos_out
        #return rgb_img, pos_out
        #return rgb_img, [pos_out, width_out]
        return rgb_img, output
        
