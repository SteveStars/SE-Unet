# -*- coding = utf-8
# @Author: Li yongquan: 1668767451@qq.com
# @Time:2022/6/15-16:41
# @File:test.py
# @Software:PyCharm

from utils import *
from utils_hcp import *
from model.Unet1_24 import unet

# totalMemory: 15.90GiB
os.environ["CUDA_VISIBLE_DEVICES"] = '3'  # '1':60  # '1':30  # '0':20  # '0':90  # '2': 40
subjects = ["111211", "127226", "206929", "902242"]


def main(data_type):
    # 90 + 10 = 4.203GiB
    train_num = 900  # 90  # 60  # 30
    test_num = 0  # 100  # 10  # 10 # 10
    # # dice sigmoid 1
    # 1. normal
    checkpoint_file = "/storage/c_lyq/plugs/happy/save_model/train_on_" + str(
        train_num) + "_person_test_on_" + str(test_num) + "_person/" + data_type + "_model.h5"  # ！！！ need change before each train
    # outputs_dir_name = "outputs_train_on_" + str(train_num) + "_person_test_on_" + str(test_num) + "_person"
    # 2. dice
    # checkpoint_file = "/storage/c_lyq/plugs/happy/save_model/dice_sigmoid_train_on_" + str(
    #     train_num) + "_person_test_on_" + str(test_num) + "_person/" + data_type + "_model.h5"
    # outputs_dir_name = "dice_sigmoid_outputs_train_on_" + str(train_num) + "_person_test_on_" + str(test_num) + "_person"
    # 3. zoom only sulc feature, checkpoint_file use normal
    outputs_dir_name = "zoom_outputs_train_on_" + str(train_num) + "_person_test_on_" + str(test_num) + "_person"
    # #
    model = unet(T1_shape)
    model.load_weights(checkpoint_file)

    test_data, origin_test_label_data = get_test_data_hcp(data_type, subjects)
    prediction = model.predict(test_data, verbose=1)  # , batch_size=batch_size, verbose=1 numpy: (-1, 64, 64, 2)
    # print("prediction.shape:", prediction.shape)
    prediction = prediction.reshape(-1, half_num_hcp * 2, 2)  # numpy: (100, 331776, 2)
    prediction = np.argmax(prediction, axis=2)  # numpy: (100, 331776)
    prediction = prediction.reshape(-1, half_num_hcp * 2, 1)  # numpy: (100, 331776, 1)
    # print("prediction.shape:", prediction.shape)

    # # when need vision
    save_prediction_hcp(prediction, outputs_dir_name, path_of_test_l_hcp, path_of_test_r_hcp, data_type, subjects)
    print_save_overall_metrics(origin_test_label_data, prediction, checkpoint_file, data_type)

    mean_shift_hcp(data_type, outputs_dir_name, subjects)


if __name__ == '__main__':
    data_types = ["sulc_sorted_by_distance"]
    # data_types = ["area_sorted_by_distance", "curv_sorted_by_distance", "sulc_sorted_by_distance", "thickness_sorted_by_distance",
    #               "volume_sorted_by_distance"]
    for now_data_type in data_types:
        print()
        print(now_data_type)
        print()
        main(now_data_type)
