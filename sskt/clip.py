import numpy as np


def clip(image):
    """
        Normalize image from R_+ to [0, 1].

        For each channel, clip any value larger than mu + 3sigma,
        where mu and sigma are the channel mean and standard deviation.

        Input:
            image - (c, h, w) | (h, w) image array in R_+
        Output:
            image - (c, h, w) image array normalized within [0, 1]
    """
    if image.ndim == 2:
        image = image[None, ...]
    temp = np.reshape(image, (image.shape[0], -1))

    limit_max = np.mean(temp, 1) + 3.0 * np.std(temp, 1)
    limit_min = np.mean(temp, 1) - 3.0 * np.std(temp, 1)
    for i, limit in enumerate(limit_max):
        channel = temp[i, :]
        ch_min = max(limit_min[i], 0)
        channel = np.clip(channel, ch_min, limit_max[i])
        ma, mi = np.max(channel), ch_min
        channel = (channel - mi) / (ma - mi)
        temp[i, :] = channel

    return np.reshape(temp, image.shape)


if __name__ == "__main__":
    a = np.random.randn(3, 5, 5)
    a = clip(a)

