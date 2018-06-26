import matplotlib.pyplot as plt
import numpy as np




#
# def to_single_img(data):
#     # img = vutils.make_grid(roll[0, :, :, :], normalize=True, scale_each=True)
#
#     d = img.reshape(1, -1, 197)[0, :, :]
#
#     plt.imshow(d)
#     plt.hlines([52 * x for x in range(10)], 0, 196)
#     plt.show()
#


def pad_along_axis(array: np.ndarray, target_length, axis=0):
    pad_size = target_length - array.shape[axis]
    axis_nb = len(array.shape)

    if pad_size < 0:
        return array.take(indices=range(target_length), axis=axis)

    npad = [(0, 0) for _ in range(axis_nb)]
    npad[axis] = (0, pad_size)

    b = np.pad(array, pad_width=npad, mode='constant', constant_values=0)

    return b