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

class Image:
    """
    Base functions for reading image
    """
    def __init__(self, img):
        self.img = img

    def crop(self, top_left, bottom_right):
        """
        Crop the image to a bounding box given by top left and bottom right pixels.
        :param top_left: tuple, top left pixel.
        :param bottom_right: tuple, bottom right pixel
        :param resize: If specified, resize the cropped image to this size
        """
        self.img = self.img[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
        #if resize is not None:
        #    self.resize(resize)

class DepthImage(Image):
    def __init__(self, img):
        super(DepthImage, self).__init__(img)
        self.img = img

    def __str__(self):
        return str(self.points)

    @classmethod
    def from_tiff(cls, fname):
        return cls(imread(fname))

    def normalize(self):
        self.img = self.img-self.img.mean()
        self.img = np.clip(self.img, -1, 1)

class RgbImage(Image):
    def __init__(self, img):
        self.img = img

    @classmethod
    def from_file(cls, fname):
        return cls(imread(fname))

    def normalize(self):
        self.img = (self.img-128)/255.0

def _gr_text_to_no(l, offset=(0, 0)):
    """
    Transform a single point from a Cornell file line to a pair of ints.
    :param l: Line from Cornell grasp file (str)
    :param offset: Offset to apply to point positions
    :return: Point [y, x]
    """
    x, y = l.split()
    return [int(round(float(y))) - offset[0], int(round(float(x))) - offset[1]]


class GraspRectangle():
    def __init__(self, points):
        self.points = points

    @property
    def width(self):
        """
        :return: Rectangle width (i.e. perpendicular to the axis of the grasp)
        """
        dy = self.points[0][0] - self.points[1][0]
        dx = self.points[0][1] - self.points[1][1]
        return np.sqrt(dx ** 2 + dy ** 2)

    @property
    def angle(self):
        """
        :return: Angle of the grasp to the horizontal.
        """
        dx = self.points[1, 1] - self.points[0, 1]
        dy = self.points[1, 0] - self.points[0, 0]
        return (np.arctan2(-dy, dx) + np.pi/2) % np.pi - np.pi/2

    def polygon_coords(self):
        gr_mod = np.zeros((4,2))
        # Get 1/3 of width
        gr_mod[0][0] = min(self.points[0][0] + (self.points[1][0]-self.points[0][0])/3, 299)
        gr_mod[0][1] = min(self.points[0][1] + (self.points[1][1]-self.points[0][1])/3, 299)
        gr_mod[1][0] = min(self.points[1][0] - (self.points[1][0]-self.points[0][0])/3, 299)
        gr_mod[1][1] = min(self.points[1][1] - (self.points[1][1]-self.points[0][1])/3, 299)
        gr_mod[2][0] = min(self.points[2][0] + (self.points[3][0]-self.points[2][0])/3, 299)
        gr_mod[2][1] = min(self.points[2][1] + (self.points[3][1]-self.points[2][1])/3, 299)
        gr_mod[3][0] = min(self.points[3][0] - (self.points[3][0]-self.points[2][0])/3, 299)
        gr_mod[3][1] = min(self.points[3][1] - (self.points[3][1]-self.points[2][1])/3, 299)

        rr, cc = polygon(gr_mod[:,0], gr_mod[:,1])
        return rr, cc

    def offset(self, offset):
        """
        Offset grasp rectangle
        :param offset: array [y, x] distance to offset
        """
        self.points += np.array(offset).reshape((1, 2))


class GraspRectangles(GraspRectangle):
    def __init__(self, grs=None):
        if grs:
            self.grs = grs
        else:
            self.grs = []

    def __len__(self):
        return len(self.grs)

    """
    def __getattr__(self, attr):

        #Test if GraspRectangle has the desired attr as a function and call it.

        # Fuck yeah python.
        if hasattr(GraspRectangle, attr) and callable(getattr(GraspRectangle, attr)):
            return lambda *args, **kwargs: list(map(lambda gr: getattr(gr, attr)(*args, **kwargs), self.grs))
        else:
            raise AttributeError("Couldn't find function %s in BoundingBoxes or BoundingBox" % attr)
    """

    @classmethod
    def load_from_cornell_file(cls, fname):
        """
        Load grasp rectangles from a Cornell dataset grasp file.
        :param fname: Path to text file.
        :return: GraspRectangles()
        """
        grs = []
        with open(fname) as f:
            while True:
                # Load 4 lines at a time, corners of bounding box.
                p0 = f.readline()
                if not p0:
                    break  # EOF
                p1, p2, p3 = f.readline(), f.readline(), f.readline()
                try:
                    gr = np.array([
                        _gr_text_to_no(p0),
                        _gr_text_to_no(p1),
                        _gr_text_to_no(p2),
                        _gr_text_to_no(p3)
                    ])
                    grs.append(GraspRectangle(gr))

                except ValueError:
                    # Some files contain weird values.
                    continue
        return cls(grs)

    def offset(self, offset):
        for i in range(0, self.__len__()):
            self.grs[i].offset(offset)

    @property
    def center(self):
        """
        Compute mean center of all GraspRectangles
        :return: float, mean centre of all GraspRectangles
        """
        #points = [gr.points for gr in self.grs]
        return np.array([240, 320])

    def draw(self, shape, position=True, width=True, angle=True):
        if position:
            pos_out = np.zeros(shape)
        if width:
            width_out = np.zeros(shape)
        if angle:
            cos_out = np.zeros(shape)
            sin_out = np.zeros(shape)
            angle_out = np.zeros(shape)
        for gr in self.grs:
            rr, cc = gr.polygon_coords()
            if position:
                pos_out[rr, cc] = 1.0
            if width:
                width_out[rr, cc] = gr.width/150.0
            if angle:
                cos_out[rr, cc] = np.cos(2.0*gr.angle)
                sin_out[rr, cc] = np.sin(2.0*gr.angle)
                #cos_out[rr, cc] = np.cos(gr.angle)
                #sin_out[rr, cc] = (np.sin(gr.angle)+1)/2.0
                #angle[rr, cc] =
        return pos_out, width_out, cos_out, sin_out
