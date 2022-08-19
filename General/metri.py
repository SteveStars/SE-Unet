import pandas as pd
# from write import creat_excel
import numpy
import numpy as np
from xlutils.copy import copy
import xlutils.copy
import xlwt
import xlrd
import matplotlib.pyplot as plt
import xlutils
from keras import optimizers
from keras.models import Model
from keras.optimizers import RMSprop
from keras.callbacks import LearningRateScheduler
from math import pow, floor

from keras import backend as K

import os
import tensorflow as tf

from keras import backend as K
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


# os.environ["OMP_NUM_THREADS"] = "3"  #use CPU


def Dice(y_pred, y_true):
    """Define the dice coefficient
        Args:
        y_pred: Prediction
        y_true: Ground truth Label
        Returns:
        Dice coefficient
        """

    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)

    y_pred_f = tf.nn.sigmoid(y_pred)
    y_pred_f = tf.cast(tf.greater(y_pred_f, 0.5), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred_f, [-1]), tf.float32)

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    dice = (2. * intersection) / (union + 0.00001)

    if (tf.reduce_sum(y_pred) == 0) and (tf.reduce_sum(y_true) == 0):
        dice = 1

    return dice


def Bage(y_true, y_pred):
    # for i in len(y_true):
    Bage = np.mean(y_pred - y_true)
    return Bage


def Baget(y_true, y_pred):
    Bage = y_pred - y_true
    return Bage


def R2(y_true, y_pred):
    a = K.square(y_pred - y_true)
    b = K.sum(a)
    c = K.mean(y_true)
    d = K.square(y_true - c)
    e = K.sum(d)
    f = 1 - b / e
    return f


def R2P(y_true, y_pred):
    a = np.square(y_pred - y_true)
    b = np.sum(a)
    c = np.mean(y_true)
    d = np.square(y_true - c)
    e = np.sum(d)
    f = 1 - b / e
    return f


def R2ad(y_true, y_pred, feature):
    a = np.square(y_pred - y_true)
    b = np.sum(a)
    c = np.mean(y_true)
    d = np.square(y_true - c)
    e = np.sum(d)
    f = 1 - b / e
    n = len(y_true)
    p = feature
    r2ad = 1 - ((1 - np.square(f)) * (n - 1)) / (n - p - 1)
    return r2ad


def MSE2(y_true, y_pred):
    mse_test = np.sum((y_pred - y_true) ** 2) / len(y_true)
    return mse_test


def RMSE(y_true, y_pred):
    mse_test = np.sum((y_pred - y_true) ** 2) / len(y_true)
    rmser = np.sqrt(mse_test)
    return rmser


def pearsona(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = np.mean(x, axis=0)
    my = np.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = np.sum(xm * ym)
    x_square_sum = np.sum(xm * xm)
    y_square_sum = np.sum(ym * ym)
    r_den = np.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return np.mean(r)


def pearsonA(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return K.mean(r)


def computeDice(autoSeg, groundTruth, n_classe):
    """ Returns
    -------
    DiceArray : floats array
          
          Dice coefficient as a float on range [0,1].
          Maximum similarity = 1
          No similarity = 0 """

    n_classes = n_classe
    DiceArray = []

    for c_i in range(1, n_classes):
        idx_Auto = np.where(autoSeg.flatten() == c_i)[0]
        idx_GT = np.where(groundTruth.flatten() == c_i)[0]

        autoArray = np.zeros(autoSeg.size, dtype=np.bool)
        autoArray[idx_Auto] = 1

        gtArray = np.zeros(autoSeg.size, dtype=np.bool)
        gtArray[idx_GT] = 1

        dsc = dice(autoArray, gtArray)

        DiceArray.append(dsc)
    return DiceArray


def dice(im1, im2):
    """
    Computes the Dice coefficient
    ----------
    im1 : boolean array
    im2 : boolean array
    
    If they are not boolean, they will be converted.
    
    -------
    It returns the Dice coefficient as a float on the range [0,1].
        1: Perfect overlapping 
        0: Not overlapping 
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.size != im2.size:
        raise ValueError("Size mismatch between input arrays!!!")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return 1.0

    # Compute Dice 
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


def dice_similarity(GT_img, Seg_img):
    """   
    Inputs:
        Seg_img (numpy.ndarray): Segmented Image.
        GT_img (numpy.ndarray): Ground Truth Image.
        State: "nifti" if the images are nifti file
               "arr"   if the images are an ndarray
    output:
        Dice Similarity Coefficient: dice_CSF, dice_GM, dice_WM."""

    segmented_data = Seg_img.copy()
    groundtruth_data = GT_img.copy()

    # Calculte DICE
    def dice_coefficient(SI, GT):
        #   2 * TP / (FN + (2 * TP) + FP)
        intersection = np.logical_and(SI, GT)
        return 2. * intersection.sum() / (SI.sum() + GT.sum())

    # Dice  for CSF
    Seg_BG = (segmented_data == 0) * 1
    GT_BG = (groundtruth_data == 0) * 1
    dice_BG = dice_coefficient(Seg_BG, GT_BG)
    # Dice  for CSF
    Seg_CSF = (segmented_data == 1) * 1
    GT_CSF = (groundtruth_data == 1) * 1
    dice_CSF = dice_coefficient(Seg_CSF, GT_CSF)
    """#Dice  for GM
    Seg_GM = (segmented_data == 2) * 1
    GT_GM = (groundtruth_data == 2) * 1
    dice_GM = dice_coefficient(Seg_GM, GT_GM)
    #Dice  for GM
    Seg_Zero = (segmented_data == 3) * 1
    GT_Zero = (groundtruth_data == 3) * 1
    dice_Zero = dice_coefficient(Seg_Zero, GT_Zero)
    """

    return dice_CSF, dice_BG  # ,dice_Zero, dice_GM,
