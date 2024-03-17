import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from skimage.transform import resize


def irregular_circle(scale=500):
    """
    生成不规则的mask(0,1二值图),
    """
    figure = plt.figure(figsize=(1, 1))
    t = np.linspace(0, 2 * np.pi, scale)
    x = np.cos(t)
    y = np.sin(t)

    # h = 0.5 * np.random.rand(scale) * np.cos(t * np.random.randint(2, 10, size=scale))
    # amp = np.sort(np.random.randn(scale))
    coeff = np.repeat(np.random.randint(2, 10, size=scale//100), 100, axis=None)
    h = 0.3 * np.cos(t * coeff)
    x += h * np.cos(t)
    y += h * np.sin(t)

    plt.axis('off')
    plt.fill(x, y, 'black')
    plt.margins(0, 0)
    figure.subplots_adjust(0, 0, 1, 1)

    figure.canvas.draw()
    w, h = figure.canvas.get_width_height()
    buf = np.fromstring(figure.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)
    image = 255 - image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('ee', image)
    # cv2.waitKey()
    # cv2.imwrite('ee.png', image)
    plt.close()

    return image // 255


def gen_mask(size=(1000, 500)):
    mask = np.zeros(shape=size)
    iter = 50
    while iter > 0:
        x = random.randint(0, size[0])
        y = random.randint(0, size[1])
        width = random.randint(20, min(size)//3)
        tmp_x = x + width - size[0]
        tmp_y = y + width - size[1]
        tmp_x = tmp_x if tmp_x > 0 else 0
        tmp_y = tmp_y if tmp_y > 0 else 0

        width -= max(tmp_x, tmp_y)

        gen_coord = (x, x + width, y, y + width)

        paste_op = random.randint(0, 2)
        if paste_op == 0:
            paste_mask = np.ones((width, width))
        elif paste_op == 1:
            mesh_a = np.arange(0, width)
            mesh = np.meshgrid(mesh_a, mesh_a)
            mesh = np.stack(mesh, axis=0)
            circle_x = (width - 1) // 2
            center_mesh = np.array([[[circle_x, ]], [[circle_x, ]]])
            dist = np.linalg.norm(mesh - center_mesh, axis=0)
            paste_mask = np.zeros((width, width))
            paste_mask[dist <= circle_x] = 1.0
        else:
            # 生成不规则mask
            paste_mask = irregular_circle()
            paste_mask = resize(paste_mask, (width, width), preserve_range=True)
            paste_mask = np.where(paste_mask < 0.5, 0, 1)

        tmp = mask[gen_coord[0]: gen_coord[1], gen_coord[2]: gen_coord[3]]
        mask[gen_coord[0]: gen_coord[1], gen_coord[2]: gen_coord[3]] \
            = np.logical_xor(tmp, paste_mask).astype(np.float)

        iter -= 1

    mask = mask.astype(np.uint8) * 255
    cv2.imshow("eeee", mask)
    cv2.waitKey()


if __name__ == "__main__":
    # irregular_circle()
    gen_mask()
