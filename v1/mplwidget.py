import numpy as np
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.ticker import EngFormatter

class MplWidget(QtWidgets.QWidget):
    
    def __init__(self, parent = None):

        QtWidgets.QWidget.__init__(self, parent)
        
        self.canvas = FigureCanvas(Figure(tight_layout=True))
        self.mpl_toolbar = NavigationToolbar2QT(self.canvas, QtWidgets.QWidget())

        vertical_layout = QtWidgets.QVBoxLayout()
        vertical_layout.addWidget(self.canvas)
        vertical_layout.addWidget(self.mpl_toolbar)

        self.setLayout(vertical_layout)
        
        
        self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.setup_axes()
        # self.canvas.axes.xaxis.set_ticks([20,100,1000,10000,20000])
        
    def setup_axes(self):
        self.canvas.axes.set_title('Aislamiento de ruido aéreo de un panel simple')
        self.canvas.axes.set_xscale('log')
        self.canvas.axes.grid(which='both')
        formatter0 = EngFormatter(unit='Hz',sep="\N{THIN SPACE}")
        formatter1 = EngFormatter(unit='dB',sep="\N{THIN SPACE}")
        self.canvas.axes.xaxis.set_major_formatter(formatter0)        
        self.canvas.axes.yaxis.set_major_formatter(formatter1)        
        self.canvas.axes.set_xlabel('Frecuencias')
        self.canvas.axes.set_ylabel('Aislamiento')
        self.canvas.axes.set_ylim(0,100)
        self.canvas.axes.set_xlim(20,20000)
        self.canvas.axes.set_xticks(np.array([31.5,63,125,250,500,
                                              1000,2000,4000,8000,16000]))