import uuid
import os

import ujson


class FileStore(object):
    data_dir = './checkpoints'

    def __init__(self, data_dir=None):
        if data_dir:
            self.data_dir = data_dir

        self.store_id = uuid.uuid4()

    @property
    def store_path(self):
        return f'{self.data_dir}/{self.store_id}'

    def prepare(self):
        """Creates model checkpoint directory
        """

        os.makedirs(self.store_path, exist_ok=True)
        self.sequence = []

    def save(self, engine):
        """Make model snapshot as json and save
        """

        json_id = uuid.uuid4()
        with open(f'{self.store_path}/{json_id}.json', 'w') as f:
            json_model = ujson.dumps(engine.model.map.tolist())
            f.write(json_model)

        self.sequence.append(str(json_id))
        with open(f'{self.store_path}/index.json', 'w') as f:
            f.write(ujson.dumps(self.sequence))
