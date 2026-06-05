import logging
import os
import tqdm


# from https://stackoverflow.com/questions/38543506/change-logging-print-function-to-tqdm-write-so-logging-doesnt-interfere-wit
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


class MkdirFileHandler(logging.FileHandler):
    def __init__(self, filename, mode="a", encoding=None, delay=False):
        log_dir = os.path.dirname(os.path.abspath(filename))
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        super().__init__(filename, mode, encoding, delay)
