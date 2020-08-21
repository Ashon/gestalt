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
        distance_matrix = np.subtract(self.partial_coords, bmu_coord)
        end_time = time.time() - start_time
        LOG.debug((
            'distance_matrix created'
            f' [elapsed={end_time * 1000:.0f} ms]'
        ))

        return distance_matrix

    def get_squared_dist_matrix(self, dist_matrix):
        start_time = time.time()
        squared_dist_matrix = np.multiply(
            dist_matrix, dist_matrix).sum(axis=2)
        end_time = time.time() - start_time
        LOG.debug((
            'squared_distance_matrix created'
            f' [elapsed={end_time * 1000:.0f} ms]'
        ))

        return squared_dist_matrix

    def get_activation_map(self, sqr_dist_matrix, gain):
        # activation_map = np.multiply(
        #     np.exp(np.divide(-squared_dist_matrix, squared_gain)),
        #     1 / np.sqrt(2 * np.pi)
        # )

        start_time = time.time()
        activation_map = np.exp(
            np.divide(-sqr_dist_matrix, gain ** 2))
        end_time = time.time() - start_time
        LOG.debug((
            'activation_map created'
            f' [elapsed={end_time * 1000:.0f} ms]'
        ))

        return activation_map

    def get_error_map(self, feature_vector):
        start_time = time.time()
        feature_error_map = np.subtract(feature_vector, self.partial_model)
        end_time = time.time() - start_time
        LOG.debug(f'error_map created [elapsed={end_time * 1000:.0f} ms]')

        return feature_error_map

    def train_feature_vector(self, feature_vector, bmu_coord, gain,
                             learning_rate, learn_threshold):
        LOG.debug('train vector')

        distance_matrix = self.get_dist_matrix(bmu_coord)
        squared_dist_matrix = self.get_squared_dist_matrix(distance_matrix)
        activation_map = self.get_activation_map(squared_dist_matrix, gain)
        feature_error_map = self.get_error_map(feature_vector)

        LOG.debug((
            'start train'
            f' [{bmu_coord=}]'
            f' [{gain=:.3f}]'
        ))
        start_time = time.time()
        trained_count = 0
        width, height, _ = self.partial_model.shape
        for x in np.arange(width):
            for y in np.arange(height):
                activate = activation_map[x][y] * learning_rate
                if activate >= learn_threshold:
                    trained_count += 1
                    bonus_weights = np.multiply(
                        feature_error_map[x][y], activate)
                    self.partial_model[x][y] = np.clip(
                        np.add(self.partial_model[x][y], bonus_weights),
                        a_min=0, a_max=1)

        end_time = time.time() - start_time
        LOG.debug((
            'trained'
            f' [{trained_count=} units]'
            f' [elapsed={end_time * 1000:.0f} ms]'
        ))

        return trained_count

    def prepare(self, result_queue):
        self.queue = multiprocessing.Queue()

        def _do_consume():
            while True:
                args = self.queue.get()
                result = self.train_feature_vector(*args)
                # LOG.info('Train Finished')
                result_queue.put(result)

        self.process = multiprocessing.Process(target=_do_consume)

    def start(self):
        self.process.start()

    def close(self):
        self.process.terminate()
        self.process.join()
        self.queue.close()
