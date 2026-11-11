"""Lazy M1 row sequences: simulator tensors transfer only when a host reads rows."""

from collections.abc import Sequence

from simulators.cube_to_plate_batch_scoring import STOP_REASONS


class ErrorRows(Sequence):
    def __init__(self, errors):
        self.errors = errors.clone()

    def __len__(self):
        return len(self.errors)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        return 'physics_execution_error' if bool(self.errors[index]) else None


class ScoringRows(Sequence):
    def __init__(self, fields):
        self.fields = fields

    def __len__(self):
        return len(self.fields['strict'])

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        result = {key: value[index].item() for key, value in self.fields.items()}
        result['reason'] = STOP_REASONS[result['reason']]
        return result


class OutcomeRows(Sequence):
    def __init__(self, fields, done, errors):
        self.scoring = ScoringRows(fields)
        self.done, self.errors = done.clone(), errors.clone()

    def __len__(self):
        return len(self.done)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if bool(self.errors[index]):
            return {'strict': False, 'lenient': False, 'execution_status': 'execution_error'}
        if bool(self.done[index]):
            s = self.scoring[index]
            return {'strict': s['strict'], 'lenient': s['lenient'], 'execution_status': 'complete'}
        return None
