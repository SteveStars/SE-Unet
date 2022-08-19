# -*- coding = utf-8
# @Author: Li yongquan: 1668767451@qq.com
# @Time:2022/4/29-9:43
# @File:mean_shift.py
# @Software:PyCharm

import os
import numpy as np
import pandas as pd
import sklearn.cluster as sc


def mean_shift(data_type, outputs_dir_name, test_num):
    if not isinstance(data_type, str):
        data_type = "_".join(data_type)
    start_id = 901
    bandwidth = 6.5
    directions = ["lh", "rh"]
    path_of_data = '/storage/c_lyq/plugs/happy/data/feature/test'
    for i in range(start_id, start_id + test_num):
        for direction in directions:
            position = pd.read_csv(os.path.join(path_of_data, direction, "data" + str(i), "position_sorted_by_distance.csv"),
                                   header=None)
            prediction = pd.read_csv(os.path.join(path_of_data, direction, "data" + str(i), outputs_dir_name, data_type + "_prediction.csv"),
                                     header=None)
            pos_pred = pd.merge(position, prediction, how='inner', left_index=True, right_index=True)  # dataframe: (165888, 4)
            hinge3_region = pos_pred[pos_pred[pos_pred.columns[-1]] == 1]  # dataframe: (hinge3_region_num, 4)
            if hinge3_region.empty:
                continue
            hinge3_region_position = hinge3_region.drop(pos_pred.columns[-1], axis=1)  # dataframe: (hinge3_region_num, 3)
            hinge3_region_position_df = pd.DataFrame(hinge3_region_position)
            hinge3_region_position_df.to_csv(
                os.path.join(path_of_data, direction, "data" + str(i), outputs_dir_name, data_type + "_hinge3_region.csv"),
                index=False, header=False)
            hinge3_region_position = np.array(hinge3_region_position)  # numpy: (hinge3_region_num, 3)
            mean_shift_model = sc.MeanShift(bandwidth=bandwidth, bin_seeding=True)
            mean_shift_model.fit(hinge3_region_position)
            hinge3_centroid_position = mean_shift_model.cluster_centers_  # numpy: (hinge3_centroid_num, 3)
            hinge3_centroid_position_df = pd.DataFrame(hinge3_centroid_position)
            hinge3_centroid_position_df.to_csv(
                os.path.join(path_of_data, direction, "data" + str(i), outputs_dir_name, data_type + "_hinge3_centroid.csv"),
                index=False, header=False)


if __name__ == '__main__':
    mean_shift("area", 10)
