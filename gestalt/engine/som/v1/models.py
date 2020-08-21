import logging
import time

import numpy as np

from gestalt.engine.som import utils


LOG = logging.getLogger(__name__)


class FeatureMap(object):
    def __init__(self, width=0, height=0, dimension=0, randomize=False):
        if randomize:
            fill_method = np.random.rand
        else:
            fill_method = np.zeros

        LOG.debug(f'generate map')
        start_time = time.time()
        self.map = fill_method(
            width * height * dimension
        ).reshape(width, height, dimension)
        end_time = time.time() - start_time
        LOG.debug((
            'map created'
            f' [elapsed={end_time * 1000:.0f} ms]'
        ))

        self._width = width
        self._height = height
        self.dimension = dimension

    def find_bmu_coord(self, feature_vector):
        ''' returns best matching unit's coord '''

        errors = np.subtract(self.map, feature_vector)
        squared_errors = np.multiply(errors, errors)
        sum_squared_errors = np.sum(squared_errors, axis=2)
        min_error = np.amin(sum_squared_errors)
        min_error_address = np.where(sum_squared_errors == min_error)

        return [min_error_address[0][0], min_error_address[1][0]]

    def find_bmu(self, feature_vector):
        ''' returns bmu '''
        bmu_coord = self.find_bmu_coord(feature_vector)

        return self.map[bmu_coord]

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height


class Som(FeatureMap):
    def __init__(self, width=0, height=0, dimension=0,
                 offset_x=0, offset_y=0, randomize=False,
                 threshold=0.5, learning_rate=0.05,
                 epoch=5, gain=2, batch=10):

        super(Som, self).__init__(
            width=width, height=height,
            dimension=dimension, randomize=randomize)

        LOG.debug(f'coord_matrix')
        start_time = time.time()
        self._coord_matrix = np.array([
            [x + offset_x, y + offset_y]
            for x in np.arange(self._width)
            for y in np.arange(self._height)
        ]).reshape(self._width, self._height, 2)
        end_time = time.time() - start_time
        LOG.debug((
            'coord_matrix created'
            f' [elapsed={end_time * 1000:.0f} ms]'
        ))

        self._learning_rate = utils.clamp(learning_rate, 0, 1)
        self._gain = gain
        self._current_epoch = 0
        self._epoch = int(epoch)

        self.set_threshold(threshold)

    def _set_learn_threshold(self):
        self._learn_threshold = self._threshold

    def set_learning_rate(self, learning_rate):
        self._learning_rate = utils.clamp(learning_rate, 0, 1)
        self._set_learn_threshold()

    @property
    def learning_rate(self):
        return self._learning_rate

    def set_threshold(self, threshold):
        self._threshold = utils.clamp(threshold, 0, 1)
        self._set_learn_threshold()

    @property
    def threshold(self):
        return self._threshold

    @property
    def learn_threshold(self):
        return self._learn_threshold

    def set_epoch(self, epoch):
        self._epoch = epoch

    @property
    def epoch(self):
        return self._epoch

    def do_progress(self):
        self._current_epoch += 1

    @property
    def progress(self):
        return float(self._current_epoch) / self._epoch

    def get_dist_matrix(self, bmu_coord):
        # LOG.debug(f'distance_matrix')
        start_time = time.time()
        distance_matrix = np.subtract(self._coord_matrix, bmu_coord)
        end_time = time.time() - start_time
        LOG.debug((
            'distance_matrix created'
            f' [elapsed={end_time * 1000:.0f} ms]'
        ))

        return distance_matrix

    def get_squared_dist_matrix(self, dist_matrix):
        # LOG.debug(f'squared_distance_matrix')
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

        # LOG.debug(f'activation_map')
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
        # LOG.debug(f'error_map')
        start_time = time.time()
        feature_error_map = np.subtract(feature_vector, self.map)
        end_time = time.time() - start_time
        LOG.debug(f'error_map created [elapsed={end_time * 1000:.0f} ms]')

        return feature_error_map

    def train_feature_vector(self, feature_vector):
        weight = 1 - self.progress
        gain = self._gain * weight
        learning_rate = self._learning_rate * weight
        learn_threashold = learning_rate * self._learn_threshold

        LOG.debug('train vector')

        bmu_coord = np.array(self.find_bmu_coord(feature_vector))
        distance_matrix = self.get_dist_matrix(bmu_coord)
        squared_dist_matrix = self.get_squared_dist_matrix(distance_matrix)
        activation_map = self.get_activation_map(squared_dist_matrix, gain)
        feature_error_map = self.get_error_map(feature_vector)

        LOG.debug((
            'start train'
            f' [{bmu_coord=}]'
            f' [{gain=:.3f}]'
            f' [{weight=:.3f}]'
        ))
        start_time = time.time()
        trained_count = 0
        for x in np.arange(self._width):
            for y in np.arange(self._height):
                activate = activation_map[x][y] * learning_rate
                if activate >= learn_threashold:
                    trained_count += 1
                    bonus_weights = np.multiply(
                        feature_error_map[x][y], activate)
                    self.map[x][y] = np.clip(
                        np.add(self.map[x][y], bonus_weights),
                        a_min=0, a_max=1)

        end_time = time.time() - start_time
        LOG.debug((
            'trained'
            f' [{trained_count=} units]'
            f' [elapsed={end_time * 1000:.0f} ms]'
        ))

        return trained_count

    def train_feature_map(self, features):
        """ Process 1 Epoch
        """

        total_trained = 0

        for feature in features:
            total_trained += self.train_feature_vector(feature)

        return total_trained

    def get_average_errors(self, feature_map):
        errors = []
        for unit in feature_map.map:
            bmu = self.find_bmu(unit)
            error = np.subtract(bmu, unit)
            dist = np.sqrt(np.sum(np.multiply(error, error)))
            errors += [dist]

        return np.average(errors)
