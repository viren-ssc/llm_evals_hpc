"""Atomic per-question records and exclusive per-model worker locks."""
import contextlib
import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.pending-')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@contextlib.contextmanager
def locked(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


class Store:
    def __init__(self, root, model):
        self.root = Path(root) / model
        self.model = model
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, qid):
        return self.root / (hashlib.sha256(qid.encode()).hexdigest() + '.json')

    def get(self, qid, fingerprint):
        path = self.path(qid)
        if not path.exists():
            return None
        record = json.loads(path.read_text())
        if (record['question_id'] != qid or record['model_id'] != self.model
                or record['fingerprint'] != fingerprint):
            raise ValueError('Checkpoint belongs to another question, model, or experiment')
        return record

    def put(self, record):
        if record['model_id'] != self.model:
            raise ValueError('Wrong model-result association')
        if (getattr(self, 'retain_truncated', False) and record['status'] == 'truncated'
                and record['truncated'] and record['generated_answer'].strip()):
            record['status'] = 'success'
            record['accepted_truncation'] = True
            record['retention_policy'] = 'Keep all generated text and continue; user override'
        if record['status'] == 'success' and (not record['generated_answer'].strip()
                or (record['truncated'] and not record.get('accepted_truncation'))):
            raise ValueError('Empty or unaccepted truncated answers cannot count as success')
        atomic(self.path(record['question_id']), record)
