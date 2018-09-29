import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as pyplot

class Graph:
    def __init__(self, ylabel, xlabel, data):
        self.ylabel = ylabel
        self.xlabel = xlabel
        self.data   = data

    def render(self):
        pyplot.ylabel(self.ylabel)
        pyplot.xlabel(self.xlabel)
        pyplot.plot(self.data)
        pyplot.show()
