# -*- coding = utf-8
# @Author: Li yongquan: 1668767451@qq.com
# @Time:2022/6/21-10:43
# @File:train_multi2.py
# @Software:PyCharm

from itertools import combinations
from part_test_multi_feature import *
from part_train_multi_feature import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

multi_num = 2
data_types = [tup for tup in list(combinations(data_types, multi_num)) if "sulc_sorted_by_distance" in tup]


def train_multi2():
    for now_data_type in data_types:
        print()
        print(now_data_type)
        print()
        train_main(now_data_type)


def test_multi2():
    for now_data_type in data_types:
        print()
        print(now_data_type)
        print()
        test_main(now_data_type)


if __name__ == '__main__':
    train_multi2()
    test_multi2()
