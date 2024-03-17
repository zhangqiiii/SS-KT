import os
import matplotlib.pyplot as plt
from itertools import chain


def collect_metric(file_dir):
    dir_list = os.listdir(file_dir)
    dir_list = list(filter(lambda x: "epoch_" in x, dir_list))
    dir_list.sort(key=lambda x: int(x.split('_')[1]))
    metrics_dic = {}
    epoch_ticks = []
    for epoch_dir in dir_list:
        now_path = os.path.join(file_dir, epoch_dir)
        if os.path.isdir(now_path):
            metric_file = os.path.join(now_path, 'metrics.txt')
            epoch = int(epoch_dir.split('_')[1])
            epoch_ticks.append(epoch)
        else:
            continue
        with open(metric_file) as f:
            lines = f.readlines()
            for i in range(0, len(lines), 2):
                lines[i] = lines[i].strip()
                value = float(lines[i+1].strip().split(' ')[1][:-1])
                if lines[i] in metrics_dic.keys():
                    metrics_dic[lines[i]].append(value)
                else:
                    metrics_dic[lines[i]] = [value]
    xticks = epoch_ticks
    data = [(xticks, metrics_dic[a]) for a in metrics_dic.keys()]

    plt.figure(figsize=(16, 7))
    for sub_data in data:
        for x, y in zip(*sub_data):
            plt.text(x, y, f'{y:.4f}', fontdict={'fontsize': 10})
    data = list(chain.from_iterable(data))
    plt.plot(*data)
    plt.ylim(0, 1)
    plt.legend(metrics_dic.keys())
    plt.savefig(os.path.join(file_dir, 'metric.png'))
    plt.close()


if __name__ == "__main__":
    collect_metric("/home/omnisky/changeDetection/uvcgan-main/result/"
                   "exp_prior_96layer_no_sparse_no_penalty_batch64_update_pseudo_l2pull/"
                   "train_yellow_0920091744")
