
import os.path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import re


def plot_loss(log_path, loss_item):
    """
    Plot loss curve.
    """
    with open(log_path, 'r') as f:
        index = 0
        index_list = []
        epoch_list = []
        loss_list = [[] for i in loss_item]
        batch = None
        for line in f.readlines():
            str_list = line.strip().split()
            index += 1
            epoch_list.append(re.split(':', str_list[0])[-1])
            if batch is None:
                batch = re.split('[()/]', str_list[1])[-2]
                batch = int(batch)
            index_list.append(index)
            for i, loss in enumerate(loss_list):
                loss.append(float(str_list[2 * i + 3]))

    fig = plt.figure(figsize=(10, 5))
    # fig = plt.figure(figsize=(20, 15))
    # fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

    # 在画布上添加一个子视图
    ax1 = fig.add_subplot()
    # zip_index_list = [index_list for i in loss_list]
    # plot_list = zip(zip_index_list, loss_list)
    # plot_list_ = []
    # for zip_item in plot_list:
    #     plot_list_.extend(zip_item)
    # ax1.plot(*plot_list_)
    for i, loss in enumerate(loss_list):
        ax1.plot(index_list, loss, label=loss_item[i])

    # ax1.xaxis.set_major_locator(MultipleLocator(batch))
    # ticks = ax1.get_xticks()
    # plt.xticks(ticks, [int(i / batch) for i in ticks])
    # plt.xticks(ticks, epoch_list)

    ax1.legend()
    fig.savefig(os.path.join(os.path.dirname(log_path), "loss.png"))
    plt.close()


if __name__ == '__main__':
    plot_loss('/home/omnisky/changeDetection/uvcgan-main/consistency_result/'
              'test_mix_contrast_mix-loss-integrate_decay3-5-8-0.1_32-64-dropout-0.5_proto/'
              'train_Italy_0417220552/train.log', ('loss', 'mix_loss'))
