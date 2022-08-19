# -*- coding = utf-8
# @Author: Li yongquan: 1668767451@qq.com
# @Time:2022/4/28-15:21
# @File:train.py
# @Software:PyCharm

from utils import *
from mean_shift import *
from keras import optimizers
from General.metri import Dice
from model.Unet1_24 import unet
from keras.callbacks import ModelCheckpoint
from General.loss_function import soft_dice_loss
from keras.callbacks import LearningRateScheduler


os.environ["CUDA_VISIBLE_DEVICES"] = '3'


def main(data_type):
    train_num = 900
    test_num = 0
    true_test_num = 100
    checkpoint_file = "/storage/c_lyq/plugs/happy/save_model/train_on_" + str(
        train_num) + "_person_test_on_" + str(test_num) + "_person/" + data_type + "_model.h5"
    outputs_dir_name = "outputs_train_on_" + str(train_num) + "_person_test_on_" + str(test_num) + "_person"
    if not os.path.exists(checkpoint_file[:checkpoint_file.rfind('/')]):
        os.makedirs(checkpoint_file[:checkpoint_file.rfind('/')])
    model = unet(T1_shape)
    if not os.path.exists(
            os.path.join(path_of_test_r + str(900 + test_num), outputs_dir_name, data_type + "_prediction.csv")):
        if not os.path.exists(checkpoint_file):  # True: #
            train_data, train_label_data = get_train_data(data_type, train_num)
            # model.summary()
            RMSprop = optimizers.RMSprop(lr=0.05, rho=0.9, epsilon=1e-4, decay=0.0)
            model.compile(optimizer=RMSprop, loss=soft_dice_loss, metrics=[Dice])
            learning_rate = LearningRateScheduler(scheduler)
            checkpoint = ModelCheckpoint(checkpoint_file, monitor='loss', verbose=1, save_best_only=True, save_weights_only=True,
                                         mode='auto')
            callbacks_list = [checkpoint, learning_rate]
            history = model.fit(train_data, train_label_data, batch_size=batch_size, epochs=epoch, callbacks=callbacks_list)

            save_history_pic_for_part_train(history, checkpoint_file, data_type)
        else:
            model.load_weights(checkpoint_file)

        test_data, origin_test_label_data = get_test_data(data_type, true_test_num)
        prediction = model.predict(test_data, verbose=1)  # numpy: (-1, 64, 64, 2)
        prediction = prediction.reshape(-1, half_num * 2, 2)  # numpy: (100, 331776, 2)
        prediction = np.argmax(prediction, axis=2)  # numpy: (100, 331776)
        prediction = prediction.reshape(-1, half_num * 2, 1)  # numpy: (100, 331776, 1)
        save_prediction(prediction, outputs_dir_name, path_of_test_l, path_of_test_r, data_type, test_num)
        print_save_overall_metrics(origin_test_label_data, prediction, checkpoint_file, data_type)

        mean_shift(data_type, outputs_dir_name, test_num)


if __name__ == '__main__':
    data_types = ["area_sorted_by_distance", "curv_sorted_by_distance", "sulc_sorted_by_distance", "thickness_sorted_by_distance",
                  "volume_sorted_by_distance"]
    for now_data_type in data_types:
        print()
        print(now_data_type)
        print()
        main(now_data_type)
