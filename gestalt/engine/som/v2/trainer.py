import logging
import multiprocessing
import time

import numpy as np

LOG = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, partial_model, partial_coords):
        self.partial_model = partial_model
        self.partial_coords = partial_coords

    def get_dist_matrix(self, bmu_coord):
        start_time = time.time()
        diff_matrix = np.subtract(self.partial_coords, bmu_coord)

        distance_matrix = np.multiply(
            diff_matrix, diff_matrix).sum(axis=2)

        elapsed = time.time() - start_time
        LOG.debug((
            'distance_matrix created'
            f' [elapsed={elapsed * 1000:.0f} ms]'
        ))

        return distance_matrix

    def get_activation_map(self, sqr_dist_matrix, gain,
                           learning_rate, learn_threshold):

        # activation_map = np.multiply(
        #     np.exp(np.divide(-squared_dist_matrix, squared_gain)),
        #     learning_rate / np.sqrt(2 * np.pi)
        # )

        start_time = time.time()
        activation_map = np.multiply(
            np.exp(np.divide(-sqr_dist_matrix, gain ** 2)),
            learning_rate
        ).reshape((*self.partial_model.shape[:2], 1))

        activation_map[activation_map <= learn_threshold] = 0

        elapsed = time.time() - start_time
        LOG.debug((
            'activation_map created'
            f' [elapsed={elapsed * 1000:.0f} ms]'
        ))

        return activation_map

    def get_activation_bound(self, activation_map):
        start_time = time.time()

        (x1, y1, x2, y2) = (0, 0, 0, 0)
        coords = np.where(activation_map > 0)

        if len(coords[0]):
            x1 = np.min(coords[0])
            x2 = np.max(coords[0]) + 1

        if len(coords[1]):
            y1 = np.min(coords[1])
            y2 = np.max(coords[1]) + 1

        bound = (x1, y1, x2, y2)
        elapsed = time.time() - start_time

        LOG.debug((
            f'activation bound [{bound=}]'
            f' [elapsed={elapsed * 1000:.0f} ms]'
        ))

        return bound

    def get_error_map(self, feature_vector):
        start_time = time.time()
        feature_error_map = np.subtract(feature_vector, self.partial_model)
        elapsed = time.time() - start_time
        LOG.debug(f'error_map created [elapsed={elapsed * 1000:.0f} ms]')

        return feature_error_map

    def train_feature_vector(self, feature_vector, bmu_coord, gain,
                             learning_rate, learn_threshold):
        LOG.debug('train vector')

        distance_matrix = self.get_dist_matrix(bmu_coord)
        activation_map = self.get_activation_map(
            distance_matrix, gain, learning_rate, learn_threshold)

        (x1, y1, x2, y2) = self.get_activation_bound(activation_map)
        trained_count = (x2 - x1) * (y2 - y1)
        if trained_count == 0:
            LOG.debug('skip train feature')
            return trained_count

        LOG.debug((
            'train feature start'
            f' [{bmu_coord=}]'
            f' [{gain=:.3f}]'
            f' [{trained_count=} units]'
        ))
        start_time = time.time()

        feature_error_map = self.get_error_map(feature_vector)
        bonus_weights = np.multiply(feature_error_map, activation_map)

        changed = np.clip(
            np.add(self.partial_model, bonus_weights),
            a_min=0, a_max=1)

        for x in np.arange(x1, x2):
            for y in np.arange(y1, y2):
                self.partial_model[x][y] = changed[x][y]

        elapsed = time.time() - start_time
        LOG.debug((
            'train feature finished'
            f' [elapsed={elapsed * 1000:.0f} ms]'
        ))

        return trained_count

    def prepare(self, result_queue):
        self.queue = multiprocessing.Queue()

        def _do_consume():
            while True:
                args = self.queue.get()
                result = self.train_feature_vector(*args)
                result_queue.put(result)

        self.process = multiprocessing.Process(target=_do_consume)

    def start(self):
        self.process.start()

    def close(self):
        self.process.terminate()
        self.process.join()
        self.queue.close()
