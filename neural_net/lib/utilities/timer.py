from time     import time
from datetime import datetime, timedelta

class Timer:
    def time(self, fn):
        start_time        = time()
        result            = fn()
        end_time          = time()
        seconds           = timedelta(seconds=int(end_time - start_time))
        self.time_elapsed = datetime(1,1,1) + seconds

        return result

    def string(self):
        time_components = [self.day(), self.hour(), self.minute(), self.second()]
        return ":".join([str(el) for el in time_components])

    def day(self):
        return self.time_elapsed.day - 1

    def hour(self):
        return self.time_elapsed.hour

    def minute(self):
        return self.time_elapsed.minute

    def second(self):
        return self.time_elapsed.second
