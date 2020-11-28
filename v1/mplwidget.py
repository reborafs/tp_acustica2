import numpy as np
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure

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
        self.canvas.axes.set_xscale('log')
        self.canvas.axes.grid(which='both')
        self.canvas.axes.set_xlabel('Frecuencias [Hz]')
        self.canvas.axes.set_ylabel('R [dB]')
        self.canvas.axes.set_ylim(0,100)
        self.canvas.axes.set_xlim(20,20000)
        self.canvas.axes.set_xticks(np.array([20,25,31.5,40,50,63,80,100,125,160,
                                             200,250,315,400,500,630,800,1000,
                                             1250,1600,2000,2500,3150,4000,5000,
                                             6300,8000,10000,12500,16000,20000]))
        self.canvas.axes.set_xticklabels(
                        ['20','25','31.5','40','50','63','80','100','125','160',
                        '200','250','315','400','500','630','800','1k',
                        '1.25k','1.6k','2k','2.5k','3.15k','4k','5k',
                        '6.3k','8k','10k','12.5k','16k','20k'], size=9, rotation=30)
