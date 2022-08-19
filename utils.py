# -*- coding = utf-8
# @Author: Li yongquan: 1668767451@qq.com
# @Time:2022/6/21-9:06
# @File:utils.py
# @Software:PyCharm

import os
import json
import keras
import numpy as np
import pandas as pd
from math import pow, floor
from General.metri3 import conf
import matplotlib.pyplot as plt
from model.Unet1_2add import unet2
from model.Unet1_3add import unet3
from model.Unet1_4add import unet4
from model.Unet1_5add import unet5

l_num = 1
k_num = 16
half_num = 165888
label_file_name = "label"  # "label_extend_2hinge"  #

epoch = 150  # 1
batch_size = 40
patch_size = 64
path_of_train_l = '/storage/c_lyq/plugs/happy/data/feature/train/lh/data'
path_of_train_r = '/storage/c_lyq/plugs/happy/data/feature/train/rh/data'
path_of_test_l = '/storage/c_lyq/plugs/happy/data/feature/test/lh/data'
path_of_test_r = '/storage/c_lyq/plugs/happy/data/feature/test/rh/data'

T1_shape = (patch_size, patch_size, k_num)


def scheduler(epoch):
    """compute learning rate"""
    init_lrate = 0.05
    drop = 0.5
    epochs_drop = 10
    lrate = init_lrate * pow(drop, floor(1 + epoch) / epochs_drop)
    print("lr changed to {}".format(lrate))
    return lrate


def load_data(path_of_train, i, d_type):
    train = os.path.join(path_of_train + str(i), d_type + ".csv")
    data = pd.read_csv(train, header=None)
    differ_num = half_num - data.shape[0]
    differ_np = np.zeros((differ_num, l_num if d_type == label_file_name else k_num))
    differ_df = pd.DataFrame(differ_np)
    data = data.append(differ_df, ignore_index=True)
    return data  # dataframe: (165888,16)


def prepare_data(path_of_l, path_of_r, is_test, d_type, total_number_of_subjects):
    all_data = []
    start_id = 901 if is_test else 1
    for i in range(start_id, start_id + total_number_of_subjects):
        if i % 10 == 1 and is_test:
            print("subject_id:", i)
        elif i % 100 == 1:
            print("subject_id:", i)
        data_l = load_data(path_of_l, i, d_type)
        data_r = load_data(path_of_r, i, d_type)
        # data_r = pd.DataFrame(np.zeros((half_num, l_num if d_type == "label" else k_num)))
        data = data_l.append(data_r, ignore_index=True)
        data = np.array(data)
        all_data.append(data)
    all_data = np.array(all_data)
    return all_data  # numpy: (total_number_of_subjects, 331776, 16) or (total_number_of_subjects, 331776, 1)


def get_test_data(data_type, true_test_num):
    origin_test_data = prepare_data(path_of_test_l, path_of_test_r, True, data_type,
                                    true_test_num)  # test_num)  # numpy: (100, 331776, 16)
    test_data = origin_test_data.reshape(-1, patch_size, patch_size, k_num)  # numpy: (-1, 64, 64, 16)
    origin_test_label_data = prepare_data(path_of_test_l, path_of_test_r, True, label_file_name,
                                          true_test_num)  # numpy: (100, 331776, 1)
    return test_data, origin_test_label_data


def get_train_data(data_type, train_num):
    origin_train_data = prepare_data(path_of_train_l, path_of_train_r, False, data_type, train_num)  # numpy: (900, 331776, 16)
    train_data = origin_train_data.reshape(-1, patch_size, patch_size, k_num)  # numpy: (-1, 64, 64, 16)
    origin_train_label_data = prepare_data(path_of_train_l, path_of_train_r, False, label_file_name,
                                           train_num)  # numpy: (900, 331776, 1)
    # print("origin_train_label_data.shape:", origin_train_label_data.shape)
    train_label_data = origin_train_label_data.reshape(-1, patch_size, patch_size, l_num)  # numpy: (-1, 64, 64, 1)
    train_label_data = keras.utils.to_categorical(train_label_data, num_classes=2)  # numpy: (-1, 64, 64, 2)
    # print("data:", train_data.nbytes + test_data.nbytes + train_label_data.nbytes + test_label_data.nbytes, "bytes")
    return train_data, train_label_data


def save_prediction(predictions, outputs_dir_name, path_of_l, path_of_r, data_type, total_number_of_subjects):
    if not isinstance(data_type, str):
        data_type = "_".join(data_type)
    start_id = 901
    for i in range(start_id, start_id + total_number_of_subjects):
        if not os.path.exists(os.path.join(path_of_l + str(i), outputs_dir_name)):
            os.makedirs(os.path.join(path_of_l + str(i), outputs_dir_name))
        if not os.path.exists(os.path.join(path_of_r + str(i), outputs_dir_name)):
            os.makedirs(os.path.join(path_of_r + str(i), outputs_dir_name))
        prediction = predictions[i - start_id]  # numpy: (331776, 1)
        prediction_l = pd.DataFrame(prediction[:half_num])
        prediction_r = pd.DataFrame(prediction[half_num:])
        prediction_l.to_csv(os.path.join(path_of_l + str(i), outputs_dir_name, data_type + "_prediction.csv"), index=False,
                            header=False)
        prediction_r.to_csv(os.path.join(path_of_r + str(i), outputs_dir_name, data_type + "_prediction.csv"), index=False,
                            header=False)


def print_save_overall_metrics(origin_test_label_data, prediction, checkpoint_file, data_type):
    if not isinstance(data_type, str):
        data_type = "_".join(data_type)
    (recall1, precision1, F11, FPR1, FNP1, TP1, FP1, FN1, TN1, acc, IoU_value, mIoU) = conf(origin_test_label_data, prediction)
    test_metrics = {
        'precision': float(precision1),
        'recall': float(recall1),
        'F1': float(F11),
        'FPR': float(FPR1),
        'FNP': float(FNP1),
        'TP:': float(TP1),
        'FP:': float(FP1),
        'FN:': float(FN1),
        'TN:': float(TN1),
        'acc:': float(acc),
        'IoU:': float(IoU_value),
        'mIoU:': float(mIoU)
    }
    test_metrics_json = json.dumps(test_metrics)
    with open(os.path.join(checkpoint_file[:checkpoint_file.rfind('/')], data_type + '_overall_test_metrics.json'), 'w+') as file:
        file.write(test_metrics_json)
    print('precision', precision1)
    print('recall', recall1)
    print('F1', F11)
    print('FPR', FPR1)
    print('FNP', FNP1)
    print('TP:', TP1)
    print('FP:', FP1)
    print('FN:', FN1)
    print('TN:', TN1)
    print('acc:', acc)
    print('IoU:', IoU_value)
    print('mIoU:', mIoU)


def print_save_every_metrics(origin_test_label_data, prediction, checkpoint_file, data_type):
    test_metrics = {}
    max_precision, max_precision_id = 0, 901
    max_recall, max_recall_id = 0, 901
    max_f1, max_f1_id = 0, 901
    for subject_i in range(prediction.shape[0]):
        (recall1, precision1, F11, FPR1, FNP1, TP1, FP1, FN1, TN1, acc, IoU_value, mIoU) = conf(origin_test_label_data[subject_i],
                                                                                                prediction[subject_i])
        if max_precision < float(precision1):
            max_precision = float(precision1)
            max_precision_id = subject_i + 901
        if max_recall < float(recall1):
            max_recall = float(recall1)
            max_recall_id = subject_i + 901
        if max_f1 < float(F11):
            max_f1 = float(F11)
            max_f1_id = subject_i + 901
        subject_i_metrics = {
            'precision': float(precision1),
            'recall': float(recall1),
            'F1': float(F11),
            'FPR': float(FPR1),
            'FNP': float(FNP1),
            'TP:': float(TP1),
            'FP:': float(FP1),
            'FN:': float(FN1),
            'TN:': float(TN1),
            'acc:': float(acc),
            'IoU:': float(IoU_value),
            'mIoU:': float(mIoU)
        }
        test_metrics[subject_i] = subject_i_metrics
        print('subject_i', subject_i + 901)
        print('precision', precision1)
        print('recall', recall1)
        print('F1', F11)
        print('FPR', FPR1)
        print('FNP', FNP1)
        print('TP:', TP1)
        print('FP:', FP1)
        print('FN:', FN1)
        print('TN:', TN1)
        print('acc:', acc)
        print('IoU:', IoU_value)
        print('mIoU:', mIoU)
        print()
    max_test_metrics = {
        'max_precision': max_precision,
        'max_precision_id': max_precision_id,
        'max_recall': max_recall,
        'max_recall_id': max_recall_id,
        'max_f1': max_f1,
        'max_f1_id': max_f1_id,
    }
    print(max_test_metrics)
    test_metrics_json = json.dumps(test_metrics)
    max_conf_json = json.dumps(max_test_metrics)
    with open(os.path.join(checkpoint_file[:checkpoint_file.rfind('/')], data_type + '_test_metrics.json'), 'w+') as file:
        file.write(test_metrics_json)
    with open(os.path.join(checkpoint_file[:checkpoint_file.rfind('/')], data_type + '_max_test_metrics.json'), 'w+') as file:
        file.write(max_conf_json)


def save_history_pic(history, checkpoint_file, data_type):
    if not isinstance(data_type, str):
        data_type = "_".join(data_type)
    plt.plot(history.history['Dice'])
    plt.plot(history.history['val_Dice'])
    plt.title('Model Dice')
    plt.ylabel('Dice')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(os.path.join(checkpoint_file[:checkpoint_file.rfind('/')], data_type + '_dice.png'))
    plt.close()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(os.path.join(checkpoint_file[:checkpoint_file.rfind('/')], data_type + '_loss.png'))
    plt.close()


def save_history_pic_for_part_train(history, checkpoint_file, data_type):
    if not isinstance(data_type, str):
        data_type = "_".join(data_type)
    plt.plot(history.history['Dice'])
    plt.title('Model Dice')
    plt.ylabel('Dice')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.savefig(os.path.join(checkpoint_file[:checkpoint_file.rfind('/')], data_type + '_dice.png'))
    plt.close()

    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.savefig(os.path.join(checkpoint_file[:checkpoint_file.rfind('/')], data_type + '_loss.png'))
    plt.close()


def get_model(data_type):
    if len(data_type) == 2:
        model = unet2(T1_shape, T1_shape)
    elif len(data_type) == 3:
        model = unet3(T1_shape, T1_shape, T1_shape)
    elif len(data_type) == 4:
        model = unet4(T1_shape, T1_shape, T1_shape, T1_shape)
    elif len(data_type) == 5:
        model = unet5(T1_shape, T1_shape, T1_shape, T1_shape, T1_shape)
    else:
        assert False, "特征数量有误"
    return model
