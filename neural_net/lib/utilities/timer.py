from time     import time
from datetime import datetime, timedelta

class Timer:
    def time(self, fn):
        start_time        = time()
        self.result       = fn()
        end_time          = time()
        seconds           = timedelta(seconds=int(end_time - start_time))
        self.time_elapsed = datetime(1,1,1) + seconds

    def string(self):
        return ":".join([str(el) for el in [self.day, self.hour, self.minute, self.second]])

    def day(self):
        return self.time_elapsed.day - 1

    def hour(self):
        return self.time_elapsed.hour

    def minute(self):
        return self.time_elapsed.minute

    def second(self):
        return self.time_elapsed.second
