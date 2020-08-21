import multiprocessing
import logging

import numpy as np

LOG = logging.getLogger(__name__)


class Model(object):
    def __init__(self, width, height, dimension, randomize=False):
        if randomize:
            fill_method = np.random.rand
        else:
            fill_method = np.zeros

        self.dimension = dimension

        # make map data to shared object
        obj_size = width * height * dimension
        init_array = np.ctypeslib.as_ctypes(fill_method(obj_size))
        shared_mem = multiprocessing.RawArray(init_array._type_, init_array)
        self.map = np.frombuffer(shared_mem).reshape(width, height, dimension)

        self.coords = np.array([
            [x, y]
            for x in np.arange(width)
            for y in np.arange(height)
        ]).reshape(width, height, 2)

    @property
    def obj_size(self):
        return np.product(self.map.shape)

    def split(self, count):
        proc_allocations = np.array_split(self.map, count, axis=0)
        coord_allocations = np.array_split(self.coords, count, axis=0)

        return zip(proc_allocations, coord_allocations)

    def find_bmu_coord(self, feature):
        errors = np.subtract(self.map, feature)

        # use 'abs' rather than '** 2'
        abs_errors = np.abs(errors)
        sum_abs_errors = np.sum(abs_errors, axis=2)
        min_error = np.amin(sum_abs_errors)
        min_error_address = np.where(sum_abs_errors == min_error)

        return [min_error_address[0][0], min_error_address[1][0]]

    def find_bmu(self, feature):
        bmu_coord = self.find_bmu_coord(feature)
        return self.map[bmu_coord[0], bmu_coord[1]]
