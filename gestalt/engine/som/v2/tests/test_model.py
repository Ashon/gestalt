import multiprocessing

import numpy as np

from gestalt.engine.som.v2.model import Model
from gestalt.engine.som.v2.trainer import Trainer


def test_zerofill_model():
    POOLSIZE = multiprocessing.cpu_count()

    RANDOMIZE = False
    WIDTH = 100
    HEIGHT = 50
    DIMENSION = 60

    model = Model(WIDTH, HEIGHT, DIMENSION, RANDOMIZE)
    assert model is not None
    assert model.obj_size == WIDTH * HEIGHT * DIMENSION

    trainers = [
        Trainer(partial_model, partial_coord)
        for partial_model, partial_coord in model.split(POOLSIZE)
    ]

    for t in trainers:
        t.partial_model += 1

    assert model.map.sum() == model.obj_size


def test_randomize_model():
    RANDOMIZE = True
    WIDTH = 100
    HEIGHT = 50
    DIMENSION = 60

    model = Model(WIDTH, HEIGHT, DIMENSION, RANDOMIZE)
    assert model is not None
    assert model.obj_size == WIDTH * HEIGHT * DIMENSION


def test_find_bmu():
    RANDOMIZE = False
    WIDTH = 100
    HEIGHT = 50
    DIMENSION = 60

    model = Model(WIDTH, HEIGHT, DIMENSION, RANDOMIZE)
    assert model is not None
    assert model.obj_size == WIDTH * HEIGHT * DIMENSION

    expected_coord = (30, 30)
    model.map[expected_coord] += 1
    assert model.map.sum() == DIMENSION

    feature = np.zeros(DIMENSION)
    feature += 1
    feature[-1] = 0

    bmu_coord = model.find_bmu_coord(feature)
    assert list(expected_coord) == bmu_coord

    bmu = model.find_bmu(feature)
    assert bmu.shape == (DIMENSION, )
    assert np.sum(bmu - feature) == 1
