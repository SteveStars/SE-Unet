# -*- coding = utf-8
# @Author: Li yongquan: 1668767451@qq.com
# @Time:2022/6/19-16:06
# @File:train_multi_feature.py
# @Software:PyCharm


from utils import *
from mean_shift import *
from keras import optimizers
from General.metri import Dice
from General.loss_function import soft_dice_loss
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler

data_types = ["area_sorted_by_distance", "curv_sorted_by_distance", "sulc_sorted_by_distance", "thickness_sorted_by_distance",
              "volume_sorted_by_distance"]


def train_main(data_type):
    train_num = 900
    test_num = 0
    checkpoint_file = "/storage/c_lyq/plugs/happy/save_model/multi_train_on_" + str(
        train_num) + "_person_test_on_" + str(test_num) + "_person/" + "_".join(
        data_type) + "_model.h5"
    if not os.path.exists(checkpoint_file[:checkpoint_file.rfind('/')]):
        os.makedirs(checkpoint_file[:checkpoint_file.rfind('/')])
    model = get_model(data_type)
    if not os.path.exists(checkpoint_file):
        train_data_dict = {}
        for d_type_i, d_type in enumerate(data_type):
            origin_train_data = prepare_data(path_of_train_l, path_of_train_r, False, d_type, train_num)  # numpy: (900, 331776, 16)
            train_data = origin_train_data.reshape(-1, patch_size, patch_size, k_num)  # numpy: (-1, 64, 64, 16)
            train_data = train_data.astype(np.float32)
            train_data_dict['input_' + str(d_type_i + 1)] = train_data
        origin_train_label_data = prepare_data(path_of_train_l, path_of_train_r, False, label_file_name,
                                               train_num)  # numpy: (900, 331776, 1)
        train_label_data = origin_train_label_data.reshape(-1, patch_size, patch_size, l_num)  # numpy: (-1, 64, 64, 1)
        train_label_data = keras.utils.to_categorical(train_label_data, num_classes=2)  # numpy: (-1, 64, 64, 2)
        RMSprop = optimizers.RMSprop(lr=0.05, rho=0.9, epsilon=1e-4, decay=0.0)
        model.compile(optimizer=RMSprop, loss=soft_dice_loss, metrics=[Dice])
        learning_rate = LearningRateScheduler(scheduler)
        checkpoint = ModelCheckpoint(checkpoint_file, monitor='loss', verbose=1, save_best_only=True, save_weights_only=True,
                                     mode='auto')
        callbacks_list = [checkpoint, learning_rate]
        history = model.fit(train_data_dict, train_label_data, batch_size=batch_size, epochs=epoch, callbacks=callbacks_list)

        save_history_pic_for_part_train(history, checkpoint_file, data_type)
