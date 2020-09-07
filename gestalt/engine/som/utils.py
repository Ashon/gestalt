import sys
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


ValueRange = namedtuple('ValueRange', 'min max')


def clamp(value, min_value, max_value):
    return min_value if value < min_value \
        else max_value if max_value < value else value


def print_map(feature_map, bmus=None):
    print('--' * feature_map.shape[0])
    for x in np.arange(feature_map.shape[0]):
        for y in np.arange(feature_map.shape[1]):
            i = clamp(feature_map[x][y], 0, 1)
            mark = '  '

            if bmus:
                try:
                    mark = bmus.index([x, y])
                    mark = f'{str(mark)[:2]:>2}'
                except Exception:
                    pass

            ansi_color_idx = 232 + int(i * 256 / 24)
            sys.stdout.write(
                f"\033[48;5;{ansi_color_idx}m{mark}\033[0m")
        sys.stdout.write('\n')
    sys.stdout.write('\n')


def draw_bmu(feature_map, bmus, ax):
    for y in np.arange(feature_map.shape[1]):
        for x in np.arange(feature_map.shape[0]):
            try:
                mark = bmus.index([x, y])
                mark = f'{str(mark)[:2]:>2}'
                pos = (y, x)
                ax.add_patch(
                    Rectangle(
                        pos, 1, 1, fill=False,
                        edgecolor='red', lw=1
                    )
                )
                ax.text(
                    *pos, f'{mark} [{x},{y}]',
                    horizontalalignment='left',
                    verticalalignment='bottom',
                )

            except ValueError:
                pass


def plot_map(feature_map, bmus=None):
    fig, ax = plt.subplots()

    c = ax.pcolor(
        np.arange(feature_map.shape[1] + 1),
        np.arange(feature_map.shape[0] + 1),
        feature_map
    )

    fig.colorbar(c, ax=ax)

    if bmus:
        draw_bmu(feature_map, bmus, ax)

    plt.show()


def print_som(som, sample_map, idx):
    bmu_idx_list = [som.find_bmu_coord(unit) for unit in sample_map.map]
    layers = np.split(som.map, som.dimension, axis=2)
    layer = np.reshape(layers[idx], (layers[idx].shape[:-1]))
    plot_map(layer, bmu_idx_list)


def print_som_error(som, sample_map, idx):
    sample = sample_map.map[idx]

    sample_error_map = np.subtract(som.map, sample)
    sample_error_map = np.multiply(sample_error_map, sample_error_map)
    sample_error_map = np.sum(sample_error_map, axis=2)

    sample_max_err = np.max(sample_error_map)
    sample_a_sim_map = np.divide(sample_error_map, sample_max_err)
    bmu_idx_list = [som.find_bmu_coord(unit) for unit in sample_map.map]

    plot_map(sample_a_sim_map, bmu_idx_list)
