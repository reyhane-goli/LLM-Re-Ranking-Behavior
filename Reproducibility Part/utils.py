import os, sys, datetime

class TeeLogger:
    def __init__(self, log_path: str):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.file = open(log_path, "w", encoding="utf-8")
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def write(self, data):
        self.file.write(data)
        self.file.flush()
        self.stdout.write(data)
        self.stdout.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        try:
            sys.stdout = self.stdout
            sys.stderr = self.stderr
            self.file.close()
        except Exception:
            pass

def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def ts_filename():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
