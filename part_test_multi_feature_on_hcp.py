# -*- coding = utf-8
# @Author: Li yongquan: 1668767451@qq.com
# @Time:2022/6/19-16:06
# @File:train_multi_feature.py
# @Software:PyCharm

from utils import *
from utils_hcp import *


data_types = ["area_sorted_by_distance", "curv_sorted_by_distance", "sulc_sorted_by_distance", "thickness_sorted_by_distance",
              "volume_sorted_by_distance"]
subjects = ["111211", "127226", "206929", "902242"]


def test_main(data_type):
    train_num = 900
    test_num = 0
    true_test_num = 100
    checkpoint_file = "/storage/c_lyq/plugs/happy/save_model/multi_train_on_" + str(
        train_num) + "_person_test_on_" + str(test_num) + "_person/" + "_".join(
        data_type) + "_model.h5"
    outputs_dir_name = "multi_outputs_train_on_" + str(train_num) + "_person_test_on_" + str(test_num) + "_person"
    if not os.path.exists(checkpoint_file[:checkpoint_file.rfind('/')]):
        os.makedirs(checkpoint_file[:checkpoint_file.rfind('/')])

    model = get_model(data_type)
    model.load_weights(checkpoint_file)
    if not os.path.exists(
            os.path.join(path_of_test_r_hcp + str(900 + true_test_num), outputs_dir_name, "_".join(data_type) + "_prediction.csv")):
        test_data_dict = {}
        for d_type_i, d_type in enumerate(data_type):
            origin_test_data = prepare_data_hcp(path_of_test_l_hcp, path_of_test_r_hcp, d_type, subjects)  # numpy: (100, 331776, 16)
            test_data = origin_test_data.reshape(-1, patch_size, patch_size, k_num)  # numpy: (-1, 64, 64, 16)
            test_data_dict['input_' + str(d_type_i + 1)] = test_data
        origin_test_label_data = prepare_data_hcp(path_of_test_l_hcp, path_of_test_r_hcp, label_file_name,
                                                  subjects)  # numpy: (100, 331776, 1)
        prediction = model.predict(test_data_dict, verbose=1)  # numpy: (-1, 64, 64, 2)
        prediction = prediction.reshape(-1, half_num_hcp * 2, 2)  # numpy: (100, 331776, 2)
        prediction = np.argmax(prediction, axis=2)  # numpy: (100, 331776)
        prediction = prediction.reshape(-1, half_num_hcp * 2, 1)  # numpy: (100, 331776, 1)
        save_prediction_hcp(prediction, outputs_dir_name, path_of_test_l_hcp, path_of_test_r_hcp, data_type, subjects)
        print_save_overall_metrics(origin_test_label_data, prediction, checkpoint_file, data_type)

        mean_shift_hcp(data_type, outputs_dir_name, subjects)
