import logging
import multiprocessing
import time

import numpy as np

from gestalt.engine.som.v2.trainer import Trainer


LOG = logging.getLogger(__name__)


class Engine(object):
    def __init__(self, model, pool_size):
        self.model = model

        self.trainers = [
            Trainer(partial_model, partial_coord)
            for partial_model, partial_coord in self.model.split(pool_size)
        ]

    def start(self):
        self.result_queue = multiprocessing.Queue()

        for trainer in self.trainers:
            trainer.prepare(self.result_queue)
            trainer.start()

    def close(self):
        [t.close() for t in self.trainers]

        self.result_queue.close()

    def train_feature(self, feature, gain, learning_rate, learn_threshold):
        start_time = time.time()
        bmu_coord = np.array(self.model.find_bmu_coord(feature))

        for t in self.trainers:
            t.queue.put((
                feature, bmu_coord, gain,
                learning_rate, learn_threshold
            ))

        results = []
        while len(results) < len(self.trainers):
            results.append(self.result_queue.get())

        trained_count = sum(results)
        elapsed = time.time() - start_time

        LOG.debug((
            f'feature trained'
            f' [{trained_count=} units]'
            f' [elapsed={elapsed * 1000:.0f} ms]'
        ))

        return trained_count
