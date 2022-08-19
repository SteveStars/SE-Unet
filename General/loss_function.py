import pandas as pd
import numpy as np
import os
import keras

import tensorflow as tf
from keras import optimizers

from math import pow, floor
import tensorflow as tf
from tensorflow.python.ops import array_ops
from keras import backend as K


def get_loss1(y_true, y_pred, end_points):
    """ pred: BxNxC,  #(?, 165888, 2)
        label: BxN,    #(?, 165888)"""
    reg_weight = 0.001
    epsilon = 1e-6
    axes = tuple(range(1, len(y_pred.shape) - 1))
    numerator = 2. * keras.backend.sum(y_pred * y_true, axes)
    denominator = keras.backend.sum(keras.backend.square(y_pred) + keras.backend.square(y_true), axes)
    soft_dice = 1 - keras.backend.mean(numerator / (denominator + epsilon))
    # Enforce the transformation as orthogonal matrix
    transform = end_points['transform']  # BxKxK

    a = transform.get_shape()[1].value
    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0, 2, 1]))
    mat_diff -= tf.constant(np.eye(a), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff)
    tf.summary.scalar('mat_loss', mat_diff_loss)
    return soft_dice + mat_diff_loss * reg_weight


def soft_dice_loss(y_true, y_pred):
    # skip the batch and class axis for calculating Dice score
    epsilon = 1e-6
    axes = tuple(range(1, len(y_pred.shape) - 1))
    numerator = 2. * K.sum(y_pred * y_true, axes)
    denominator = K.sum(K.square(y_pred) + K.square(y_true), axes)

    return 1 - K.mean(numerator / (denominator + epsilon))  # average over classes and batch


def get_loss(y_true, y_pred, end_points):
    """ pred: BxNxC,  #(?, 165888, 2)
        label: BxN,    #(?, 165888)"""
    reg_weight = 0.001
    print(y_true.shape)  # (?,?,?)
    print(y_pred.shape)  # (?, 165888, 2)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)

    # Enforce the transformation as orthogonal matrix
    transform = end_points['transform']  # BxKxK

    a = transform.get_shape()[1].value
    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0, 2, 1]))
    mat_diff -= tf.constant(np.eye(a), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff)
    tf.summary.scalar('mat_loss', mat_diff_loss)

    return classify_loss + mat_diff_loss * reg_weight


def point_loss(end_points):
    def get_l(y_true, y_pred):
        return get_loss1(y_true, y_pred, end_points)

    return get_l


def soft_dice_lossA(y_true, y_pred, end_points, reg_weight=0.001):
    # skip the batch and class axis for calculating Dice score
    epsilon = 1e-6
    axes = tuple(range(1, len(y_pred.shape) - 1))
    numerator = 2. * K.sum(y_pred * y_true, axes)
    denominator = K.sum(K.square(y_pred) + K.square(y_true), axes)
    soft_dice = 1 - K.mean(numerator / (denominator + epsilon))

    # matrix 64*64

    return 1 - K.mean(numerator / (denominator + epsilon))  # average over classes and batch


# parameter for loss function
def cross_entropy(targets, predictions, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions. 
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray        
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions + 1e-9)) / N
    return ce


def CE_dice_loss(y_true, y_pred):
    epsilon = 1e-12
    predictions = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -tf.reduce_sum(y_true * tf.log(predictions + 1e-9)) / N
    epsilona = 1e-6
    axes = tuple(range(1, len(y_pred.shape) - 1))
    numerator = 2. * K.sum(y_pred * y_true, axes)
    denominator = K.sum(K.square(y_pred) + K.square(y_true), axes)
    dice = 1 - K.mean(numerator / (denominator + epsilona))
    add = ce + dice
    return add


def focal_loss2(y_true, y_pred):
    gamma = 1
    alpha = .25
    gamma = float(gamma)
    alpha = float(alpha)
    epsilon = 1.e-9
    y_true = tf.convert_to_tensor(y_true, tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, tf.float32)
    model_out = tf.add(y_pred, epsilon)
    ce = tf.multiply(y_true, -tf.log(model_out))
    weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
    f1 = tf.multiply(alpha, tf.multiply(weight, ce))
    reduced_f1 = tf.reduce_max(f1, axis=1)
    focal_loss_fixed = tf.reduce_mean(reduced_f1)
    return focal_loss_fixed


def Focal_dice(y_true, y_pred):
    gamma = 2
    alpha = .25
    gamma = float(gamma)
    alpha = float(alpha)
    epsilon = 1.e-9
    y_true = tf.convert_to_tensor(y_true, tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, tf.float32)
    model_out = tf.add(y_pred, epsilon)
    ce = tf.multiply(y_true, -tf.log(model_out))
    weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
    f1 = tf.multiply(alpha, tf.multiply(weight, ce))
    reduced_f1 = tf.reduce_max(f1, axis=1)
    focal_loss_fixed = tf.reduce_mean(reduced_f1)

    epsilona = 1e-6
    axes = tuple(range(1, len(y_pred.shape) - 1))
    numerator = 2. * K.sum(y_pred * y_true, axes)
    denominator = K.sum(K.square(y_pred) + K.square(y_true), axes)
    dice = 1 - K.mean(numerator / (denominator + epsilona))
    add = focal_loss_fixed + dice
    return add


def focal_loss(classes_num, gamma=1., alpha=0.75, e=1.e-9):  # (classes_num=[317000,14000,500])
    # classes_num contains sample number of each classes
    def focal_loss_fixed(target_tensor, prediction_tensor):
        # 1# get focal loss with no balanced weight which presented in paper function (4)
        zeros = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)
        one_minus_p = array_ops.where(tf.greater(target_tensor, zeros), target_tensor - prediction_tensor, zeros)
        FT = -1 * (one_minus_p ** gamma) * tf.log(tf.clip_by_value(prediction_tensor, 1e-8, 1.0))

        # 2# get balanced weight alpha
        classes_weight = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)

        total_num = float(sum(classes_num))
        classes_w_t1 = [total_num / ff for ff in classes_num]
        sum_ = sum(classes_w_t1)
        classes_w_t2 = [ff / sum_ for ff in classes_w_t1]  # scale
        classes_w_tensor = tf.convert_to_tensor(classes_w_t2, dtype=prediction_tensor.dtype)
        classes_weight += classes_w_tensor

        alpha = array_ops.where(tf.greater(target_tensor, zeros), classes_weight, zeros)

        # 3# get balanced focal loss
        balanced_fl = alpha * FT
        balanced_fl = tf.reduce_mean(balanced_fl)

        # 4# add other op to prevent overfit
        # reference : https://spaces.ac.cn/archives/4493
        nb_classes = len(classes_num)
        fianal_loss = (1 - e) * balanced_fl + e * K.categorical_crossentropy(K.ones_like(prediction_tensor) / nb_classes,
                                                                             prediction_tensor)

        return fianal_loss

    return focal_loss_fixed


"""
def IoU(y_true, y_pred):
    eps=1e-6
    if np.max(y_true) == 0.0:
        return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return -K.mean( (intersection + eps) / (union + eps), axis=0)
"""


def ioU(y_true, y_pred):  # , label: int
    """
    Return the Intersection over Union (IoU) for a given label.
    Args:
      y_true: the expected y values as a one-hot
      y_pred: the predicted y values as a one-hot or softmax output
      label: the label to return the IoU for
    Returns:
      the IoU for the given label
    """
    # extract the label values using the argmax operator then
    # calculate equality of the predictions and truths to the label
    # y_true = K.cast(K.equal(K.argmax(y_true), label), K.floatx())
    # y_pred = K.cast(K.equal(K.argmax(y_pred), label), K.floatx())
    # calculate the |intersection| (AND) of the labels
    intersection = K.sum(y_true * y_pred)
    # calculate the |union| (OR) of the labels
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    # avoid divide by zero - if the union is zero, return 1
    # otherwise, return the intersection over union
    return 1 - K.switch(K.equal(union, 0), 1.0, intersection / union)


def ioU(y_true, y_pred):
    """
    Return the Intersection over Union (IoU) score.
    Args:
    y_true: the expected y values as a one-hot
    y_pred: the predicted y values as a one-hot or softmax output
    Returns:
     the scalar IoU value (mean over all labels)
    """
    # get number of labels to calculate IoU for
    num_labels = K.int_shape(y_pred)[-1] - 1
    # initialize a variable to store total IoU in
    mean_iou = K.variable(0)

    # iterate over labels to calculate IoU for
    for label in range(num_labels):
        mean_iou = mean_iou + iou(y_true, y_pred, label)

    # divide total IoU by number of labels to get mean IoU
    return mean_iou / num_labels


def ioU(y_true, y_pred):
    # iou loss for bounding box prediction
    # input must be as [x1, y1, x2, y2]

    # AOG = Area of Groundtruth box
    AoG = K.abs(K.transpose(y_true)[2] - K.transpose(y_true)[0] + 1) * K.abs(K.transpose(y_true)[3] - K.transpose(y_true)[1] + 1)

    # AOP = Area of Predicted box
    AoP = K.abs(K.transpose(y_pred)[2] - K.transpose(y_pred)[0] + 1) * K.abs(K.transpose(y_pred)[3] - K.transpose(y_pred)[1] + 1)

    # overlaps are the co-ordinates of intersection box
    overlap_0 = K.maximum(K.transpose(y_true)[0], K.transpose(y_pred)[0])
    overlap_1 = K.maximum(K.transpose(y_true)[1], K.transpose(y_pred)[1])
    overlap_2 = K.minimum(K.transpose(y_true)[2], K.transpose(y_pred)[2])
    overlap_3 = K.minimum(K.transpose(y_true)[3], K.transpose(y_pred)[3])

    # intersection area
    intersection = (overlap_2 - overlap_0 + 1) * (overlap_3 - overlap_1 + 1)

    # area of union of both boxes
    union = AoG + AoP - intersection

    # iou calculation
    iou = intersection / union

    # bounding values of iou to (0,1)
    iou = K.clip(iou, 0.0 + K.epsilon(), 1.0 - K.epsilon())

    # loss for the iou value
    iou_loss = -K.log(iou)

    return iou_loss


def generalized_dice_coeff(y_true, y_pred):
    Ncl = y_pred.shape[-1]
    w = K.zeros(shape=(Ncl,))
    w = K.sum(y_true, axis=(0, 1, 2))
    w = 1 / (w ** 2 + 0.000001)
    # Compute gen dice coef:
    numerator = y_true * y_pred
    numerator = w * K.sum(numerator, (0, 1, 2, 3))
    numerator = K.sum(numerator)
    denominator = y_true + y_pred
    denominator = w * K.sum(denominator, (0, 1, 2, 3))
    denominator = K.sum(denominator)
    gen_dice_coef = 2 * numerator / denominator
    return gen_dice_coef


def generalized_dice_coeff_loss(y_true, y_pred):
    return 1 - generalized_dice_coeff(y_true, y_pred)


def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    smooth = 0.2
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def IoU(y_true, y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1 - pt_1), gamma)


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard


def Iou(y_true, y_pred):  # lovasz_hinge
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    logits = y_pred
    labels = y_true
    per_image = True
    ignore = None
    if per_image:
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)

        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)

        # Fixed python3
        losses.set_shape((None,))

        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        # loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        # ELU + 1
        loss = tf.tensordot(tf.nn.elu(errors_sorted) + 1., tf.stop_gradient(grad), 1, name="loss_non_void")
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
                   strict=True,
                   name="loss"
                   )
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels


def margin_loss(y_true, y_pred):
    lamb, margin = 0.5, 0.1
    loss = y_true * K.square(K.relu((1 - margin) - y_pred)) + lamb * (1 - y_true) * K.square(K.relu(y_pred - margin))
    losses = K.sum(loss, axis=-1)  # ÕâÀï * Ä¬ÈÏÖðÔªËØÏà³Ë
    return losses
