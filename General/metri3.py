import numpy as np


# y_true = np.array([[0, 1, 0, 0, 1, 0],[1, 1, 0, 0, 1, 1]])
# y_pred = np.array([[1, 1, 1, 0, 0, 1],[1, 1, 1, 0, 0, 1]])
def conf(y_true, y_pred):
    # true positive
    TP = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 1)))

    # false positive
    FP = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 1)))

    # false negative
    FN = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 0)))

    # true negative
    TN = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 0)))

    recall = TP / (TP + FN)
    FPR = FP / (FP + TN)
    FNP = FN / (TP + FN)
    precision = TP / (TP + FP)
    F1 = (2 * (recall * precision)) / (recall + precision)

    acc = (TP + TN) / (TP + TN + FP + TN)
    IoU = TP / (TP + FP + FN)
    mIoU = np.nanmean(IoU)
    return recall, precision, F1, FPR, FNP, TP, FP, FN, TN, acc, IoU, mIoU


"""recall,precision,F1,FPR,FNP=conf(y_true, y_pred)
print(recall)
print(precision)
print(F1)
print(FPR)
print(FNP)"""
