import logging
import sys


LOG_FORMAT = ' '.join((
    '[%(asctime)s]',
    '[%(processName)s]',
    '[%(name)s]',
    '[%(levelname)s]',
    '%(message)s'
))


def get_logger(log_level):
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level, 'INFO'))

    formatter = logging.Formatter(LOG_FORMAT)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger
