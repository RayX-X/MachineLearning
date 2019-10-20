# -*- coding:utf-8 -*-
# @Time    : 2019/8/4 15:36
# @Author  : Ray.X
import sys
import time


def view_bar(step, total, loss):
    """
    这个输出方法我喜欢
    :param step: 第几次训练
    :param total: 总训练次数
    :param loss: 误差
    :return:
    """
    rate = step / total
    rate_num = int(rate * 40)
    r = '\rstep-%d loss value-%.4f[%s%s]\t%d%% %d/%d' % \
        (step, loss, '>' * rate_num, '-' * (40 - rate_num), int(rate * 100), step, total)
    sys.stdout.write(r)
    sys.stdout.flush()  # 刷新 windows 下可有可无 Lunix 下必须要有


def mode_data(max_num=50000):
    """
    假装正在训练
    :param max_num: 最大训练次数
    :return:
    """
    for i in range(max_num):
        j = (max_num - i)/max_num
        view_bar(i + 1, max_num, j)
        time.sleep(0.001)


if __name__ == "__main__":
    print("输出")
    mode_data()
