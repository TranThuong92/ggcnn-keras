from __future__ import print_function

import sys
sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")

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

from .grasp_data import GraspDatasetBase
from .axiliary import DepthImage, RgbImage, GraspRectangles
from sklearn.utils import shuffle

class cornell_data(GraspDatasetBase):
    def __init__(self, file_path, start=0.0, end=1.0, **kwargs):
        super(cornell_data, self).__init__(**kwargs)
        graspf = glob.glob(os.path.join(file_path, '*', 'pcd*cpos.txt'))
        depthf = glob.glob(os.path.join(file_path, '*', '*.tiff'))
        rgbf = glob.glob(os.path.join(file_path, '*', '*.png'))

        graspf = sorted(graspf)
        depthf = sorted(depthf)
        rgbf = sorted(rgbf)

        l = len(graspf)
        if l == 0:
            print("Datasets not found. Check path: {}".format(file_path))

        graspf, depthf, rgbf = shuffle(graspf, depthf, rgbf, random_state=20)

        self.grasp_files = graspf[int(start*l):int(end*l)]
        self.depth_files = depthf[int(start*l):int(end*l)]
        self.rgb_files = rgbf[int(start*l):int(end*l)]

    def _get_crop_attrs(self, idx):
        gtbbs = GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
        center = gtbbs.center
        left = max(0, min(center[1] - self.output_size // 2, 640 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 480 - self.output_size))
        return center, left, top

    def get_depth(self, idx):
        depth_img = DepthImage.from_tiff(self.depth_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        depth_img.crop((top, left), (min(480, top+self.output_size), min(640, left+self.output_size)))
        depth_img.normalize()
        return depth_img.img

    def get_rgb(self, idx):
        rgb_img = RgbImage.from_file(self.rgb_files[idx])
        rgb_img.normalize()
        center, left, top = self._get_crop_attrs(idx)
        rgb_img.crop((top, left), (min(480, top+self.output_size), min(640, left+self.output_size)))
        return rgb_img.img

    def get_gtbb(self, idx):
        gtbbs = GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        gtbbs.offset((-top, -left))
        return gtbbs
