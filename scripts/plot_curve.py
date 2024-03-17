import os.path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import re


def plot_pull_loss(log_path):
    with open(log_path, 'r') as f:
        index = 0
        index_rec_list = []
        index_pull_list = []
        index_dis_list = []
        rec_list = []
        pull_list = []
        dis_list = []
        sparse_list = []
        batch = None
        for line in f.readlines():
            str_list = line.strip().split(' ')
            index += 1
            if batch is None:
                batch = re.split('[()/]', str_list[0])[-2]
                batch = int(batch)
            rec = float(str_list[3])
            pull = float(str_list[5])
            dis = float(str_list[7])
            sparse = float(str_list[9])
            if rec != 0:
                index_rec_list.append(index)
                rec_list.append(rec)
            if dis != 0:
                index_dis_list.append(index)
                dis_list.append(dis)
            index_pull_list.append(index)
            pull_list.append(pull)
            sparse_list.append(sparse)

    fig = plt.figure(figsize=(20, 15))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

    # 在画布上添加一个子视图
    ax1 = fig.add_subplot(311)
    ax1.plot(index_rec_list, rec_list, color='red', label='rec')
    ax1.set_ylabel("rec_loss")
    ax1.set_xlabel("iter(epoch)")

    ax1.xaxis.set_major_locator(MultipleLocator(batch))
    ticks = ax1.get_xticks()
    plt.xticks(ticks, [int(i / batch) for i in ticks])

    ax2 = ax1.twinx()  # 很重要
    ax2.plot(index_pull_list, pull_list, label='pull')
    ax2.set_ylabel("pull_loss")

    dis = fig.add_subplot(312)
    dis.plot(index_dis_list, dis_list, color='darkgreen')
    dis.xaxis.set_major_locator(MultipleLocator(batch))
    ticks = dis.get_xticks()
    plt.xticks(ticks, [int(i / batch) for i in ticks])
    dis.set_ylabel("dis_loss")

    sparse = fig.add_subplot(313)
    sparse.plot(range(1,len(sparse_list)+1), sparse_list, color='black')
    sparse.xaxis.set_major_locator(MultipleLocator(batch))
    ticks = sparse.get_xticks()
    plt.xticks(ticks, [int(i / batch) for i in ticks])
    sparse.set_ylabel("sparse_loss")

    fig.legend(("rec", "pull", "dis", "sparse"))
    fig.savefig(os.path.join(os.path.dirname(log_path), "pull_loss.png"))
    plt.close()


if __name__ == '__main__':
    plot_pull_loss('/home/omnisky/changeDetection/uvcgan-main/result/train_Italy_0904155229/train.log')