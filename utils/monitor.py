import copy


class EarlyStopMonitor:

    def __init__(self, patient):
        self.params = None
        self.patient = patient
        self.counter = 0
        self.val = 1e10
        self.epoch = -1

    def early_stop(self):
        return self.counter >= self.patient

    def track(self, epoch, params, val):
        if val < self.val:
            self.params = params
            self.epoch = epoch
            self.val = val
            self.counter = 0
        else:
            self.counter += 1
