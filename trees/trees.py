# -*- coding:utf-8 -*-
# @Time    : 2019/10/22 19:09
# @Author  : Ray.X
from math import log

test_data = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 0, 'no']]
label = ['can live with no air ', 'have flippers']


def shannon():
    """
    熵的实现
    :return:
    """
    num = len(test_data)
    # 为所有可能的类创建字典
    lables = {}
    for vec in test_data:
        current_lable = vec[-1]
        if current_lable not in lables.keys():
            lables[current_lable] = 0
        lables[current_lable] += 1

    # 计算熵
    shan = 0.0
    for key in lables:
        pxi = float(lables[key]) / num
        shan -= pxi * log(pxi)
    return shan

