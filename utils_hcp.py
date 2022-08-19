# -*- coding = utf-8
# @Author: Li yongquan: 1668767451@qq.com
# @Time:2022/6/30-10:08
# @File:utils_hcp.py
# @Software:PyCharm

import os
import numpy as np
import pandas as pd
import sklearn.cluster as sc

l_num = 1
k_num = 16
patch_size = 64
half_num_hcp = 184320
label_file_name = "label"
path_of_test_l_hcp = '/storage/c_lyq/plugs/happy/data/feature/hcp/test/lh'
path_of_test_r_hcp = '/storage/c_lyq/plugs/happy/data/feature//hcp/test/rh'
abcd_max, abcd_min = -16.3282, 18.2493
hcp_max, hcp_min = -1.7821, 2.0155


def get_test_data_hcp(data_type, subjects):
    origin_test_data = prepare_data_hcp(path_of_test_l_hcp, path_of_test_r_hcp, data_type,
                                        subjects)  # test_num)  # numpy: (100, 331776, 16)
    test_data = origin_test_data.reshape(-1, patch_size, patch_size, k_num)  # numpy: (-1, 64, 64, 16)
    origin_test_label_data = prepare_data_hcp(path_of_test_l_hcp, path_of_test_r_hcp, label_file_name,
                                              subjects)  # numpy: (100, 331776, 1)
    return test_data, origin_test_label_data


def load_data_hcp(path_of_train, subject, d_type):
    train = os.path.join(path_of_train, subject, d_type + ".csv")
    data = pd.read_csv(train, header=None)
    # #  add @Time:2022/7/3-14:44
    if d_type == "sulc_sorted_by_distance":
        # https://zhidao.baidu.com/question/134864186335361205.html
        data = (abcd_max - abcd_min) / (hcp_max - hcp_min) * (data - hcp_min) + abcd_min
    # #
    differ_num = half_num_hcp - data.shape[0]
    differ_np = np.zeros((differ_num, l_num if d_type == label_file_name else k_num))
    differ_df = pd.DataFrame(differ_np)
    data = data.append(differ_df, ignore_index=True)
    return data  # dataframe: (165888,16)


def prepare_data_hcp(path_of_l, path_of_r, d_type, subjects):
    all_data = []
    for subject in subjects:
        print("subject:", subject)
        data_l = load_data_hcp(path_of_l, subject, d_type)
        data_r = load_data_hcp(path_of_r, subject, d_type)
        # data_r = pd.DataFrame(np.zeros((half_num_hcp, l_num if d_type == "label" else k_num)))
        data = data_l.append(data_r, ignore_index=True)
        data = np.array(data)
        all_data.append(data)
    all_data = np.array(all_data)
    return all_data  # numpy: (total_number_of_subjects, 331776, 16) or (total_number_of_subjects, 331776, 1)


def save_prediction_hcp(predictions, outputs_dir_name, path_of_l, path_of_r, data_type, subjects):
    if not isinstance(data_type, str):
        data_type = "_".join(data_type)
    for i, subject in enumerate(subjects):
        if not os.path.exists(os.path.join(path_of_l, subject, outputs_dir_name)):
            os.makedirs(os.path.join(path_of_l, subject, outputs_dir_name))
        if not os.path.exists(os.path.join(path_of_r, subject, outputs_dir_name)):
            os.makedirs(os.path.join(path_of_r, subject, outputs_dir_name))
        prediction = predictions[i]  # numpy: (331776, 1)
        prediction_l = pd.DataFrame(prediction[:half_num_hcp])
        prediction_r = pd.DataFrame(prediction[half_num_hcp:])
        prediction_l.to_csv(os.path.join(path_of_l, subject, outputs_dir_name, data_type + "_prediction.csv"), index=False,
                            header=False)
        prediction_r.to_csv(os.path.join(path_of_r, subject, outputs_dir_name, data_type + "_prediction.csv"), index=False,
                            header=False)


def mean_shift_hcp(data_type, outputs_dir_name, subjects):
    if not isinstance(data_type, str):
        data_type = "_".join(data_type)
    bandwidth = 6.5
    directions = ["lh", "rh"]
    path_of_data = '/storage/c_lyq/plugs/happy/data/feature/hcp/test'
    for subject in subjects:
        for direction in directions:
            position = pd.read_csv(os.path.join(path_of_data, direction, subject, "position_sorted_by_distance.csv"),
                                   header=None)
            prediction = pd.read_csv(os.path.join(path_of_data, direction, subject, outputs_dir_name, data_type + "_prediction.csv"),
                                     header=None)
            pos_pred = pd.merge(position, prediction, how='inner', left_index=True, right_index=True)  # dataframe: (165888, 4)
            hinge3_region = pos_pred[pos_pred[pos_pred.columns[-1]] == 1]  # dataframe: (hinge3_region_num, 4)
            if hinge3_region.empty:
                continue
            hinge3_region_position = hinge3_region.drop(pos_pred.columns[-1], axis=1)  # dataframe: (hinge3_region_num, 3)
            hinge3_region_position_df = pd.DataFrame(hinge3_region_position)
            hinge3_region_position_df.to_csv(
                os.path.join(path_of_data, direction, subject, outputs_dir_name, data_type + "_hinge3_region.csv"),
                index=False, header=False)
            hinge3_region_position = np.array(hinge3_region_position)  # numpy: (hinge3_region_num, 3)
            mean_shift_model = sc.MeanShift(bandwidth=bandwidth, bin_seeding=True)
            mean_shift_model.fit(hinge3_region_position)
            hinge3_centroid_position = mean_shift_model.cluster_centers_  # numpy: (hinge3_centroid_num, 3)
            hinge3_centroid_position_df = pd.DataFrame(hinge3_centroid_position)
            hinge3_centroid_position_df.to_csv(
                os.path.join(path_of_data, direction, subject, outputs_dir_name, data_type + "_hinge3_centroid.csv"),
                index=False, header=False)
