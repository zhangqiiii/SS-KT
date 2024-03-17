import numpy as np


def confusion_map(target_change_map, change_map):
    """
        Compute RGB confusion map for the change map.
            True positive   - White  [1,1,1]
            True negative   - Black  [0,0,0]
            False positive  - Green  [0,1,0]
            False negative  - Red    [1,0,0]
    Args
        target_change_map: predict map (1, h, w)
        change_map: ground truth (1, h, w)
    Return:
        conf_map: confusion map ndarray (3, h, w)
        metrics: tuple( kappa, recall, precision, overall_accuracy, FA, MA )
    """
    assert target_change_map.shape == change_map.shape
    conf_map = np.concatenate(
        [
            np.logical_and(target_change_map, change_map),
            target_change_map,
            change_map,
        ],
        axis=0
    )

    confuse_value = np.sum(conf_map, axis=0)
    tp = confuse_value[confuse_value == 3].shape[0]
    tn = confuse_value[confuse_value == 0].shape[0]
    fp = confuse_value[np.logical_and(confuse_value == 1, target_change_map.squeeze() == 1)].shape[0]
    fn = confuse_value[np.logical_and(confuse_value == 1, change_map.squeeze() == 1)].shape[0]

    num_ch = fn + tp
    num_unch = fp + tn
    num = num_ch + num_unch
    overall_accuracy = pra = (tp + tn)/num
    pre = ((tp + fp) * num_ch + (fn + tn) * num_unch) / (num * num)
    kappa = (pra - pre)/(1 - pre)

    FA = fp / (tn + fp)
    MA = fn / (tp + fn)

    try:
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
    except Exception:
        recall = 0
        precision = 0

    metrics = (kappa, recall, precision, overall_accuracy, FA, MA)
    return conf_map, metrics
