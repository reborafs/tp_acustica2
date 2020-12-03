import pandas as pd
import numpy as np
from process import import_material, export_to_excel, Material, octave_thirdoctave
from gui import Ui_MainWindow, TableModel
from PyQt5 import QtWidgets
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtCore import QRegExp

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self) 

        ### Inputs
        df = pd.ExcelFile('tabla_materiales.xlsx').parse()
        material_list = df['Material'].tolist()        
        for material in material_list:
            self.cb_material.addItem(material)
        thick_condition = QRegExp("\\d\\.\\d{1,4}")
        l_condition = QRegExp("\\d{1,2}\\.\\d{1,2}")
        self.input_thick.setValidator(QRegExpValidator(thick_condition))
        self.input_lx.setValidator(QRegExpValidator(l_condition))
        self.input_ly.setValidator(QRegExpValidator(l_condition))
        
        ### Buttons
        self.button_process.clicked.connect(self.process)
        self.button_clear.clicked.connect(self.clear)  
        self.button_export.clicked.connect(self.export)    
        
        ### TABLE
        self.freqs ={'octave' : [31.5,63,125,250,500,1000,2000,4000,8000,16000],
                     'third' :  [20,25,31.5,40,50,63,80,100,125,160,
                                 200,250,315,400,500,630,800,1000,
                                 1250,1600,2000,2500,3150,4000,5000,
                                 6300,8000,10000,12500,16000,20000]}
                
        self.data = pd.DataFrame([
          np.zeros(31,),
          np.zeros(31,),
          np.zeros(31,),
          np.zeros(31,),
        ], columns = self.freqs['third'], index=['Cremer', 'Sharp', 
                                                 'ISO 12354-1', 'Davy'])
        self.model = TableModel(self.data)
        self.table_R.setModel(self.model)
        
        # Set Text for fc, fd
        self.show_fc.setText('-')
        self.show_fd.setText('-')
        
    def process(self):
        material_name = self.cb_material.currentText()
        self.material = Material(import_material(material_name))
        self.thickness = float(self.input_thick.text())
        self.lx = float(self.input_lx.text())
        self.ly = float(self.input_ly.text())
        self.MplWidget.canvas.axes.clear()
        self.MplWidget.setup_axes()

        R_dict = {}
        
        if self.radio_octave.isChecked():
            self.band_type = 'octave'
            freqs = octave_thirdoctave(self.band_type)
            
        if self.radio_third.isChecked():
            self.band_type = 'third'     
            freqs = octave_thirdoctave(self.band_type)

        if self.checkbox_cremer.isChecked():
            R_cremer = self.material.cremer(self.lx,self.ly,self.thickness,self.band_type)
            self.MplWidget.canvas.axes.plot(freqs,R_cremer,label='Cremer')
            R_dict['Cremer'] = np.round(R_cremer,2)
            
        if self.checkbox_sharp.isChecked():
            R_sharp = self.material.sharp(self.lx,self.ly,self.thickness,self.band_type)
            self.MplWidget.canvas.axes.plot(freqs,R_sharp,label='Sharp')
            R_dict['Sharp'] = np.round(R_sharp,2)


        if self.checkbox_iso.isChecked():
            R_iso = self.material.iso12354(self.lx,self.ly,self.thickness,self.band_type)
            self.MplWidget.canvas.axes.plot(freqs,R_iso,label='ISO 12354')
            R_dict['ISO 12354-1'] = np.round(R_iso,2)


        if self.checkbox_davy.isChecked():
            R_davy = self.material.davy(self.lx,self.ly,self.thickness,self.band_type)
            self.MplWidget.canvas.axes.plot(freqs,R_davy,label='Davy')
            R_dict['Davy'] = np.round(R_davy,2)

        # Draw plots and write tables
        df = pd.DataFrame(R_dict).transpose()
        df.columns = freqs
        self.MplWidget.canvas.axes.legend()
        self.MplWidget.canvas.draw()
        self.table_R.clearSpans()
        self.model = TableModel(df)
        self.table_R.setModel(self.model)        
        
        # Set Text for fc, fd
        fc = self.material.freq_c(self.lx,self.ly,self.thickness)
        fd = self.material.freq_d(self.lx,self.ly,self.thickness)
        self.show_fc.setText(str(np.round(fc,2)))
        self.show_fd.setText(str(np.round(fd,2)))

    def export(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save to .xlsx", "untitled.xlsx", 
                         "Excel(*.xlsx);;All Files(*.*)") 
        if filename:
            export_to_excel(filename, self.lx,self.ly,self.thickness,
                            self.band_type,self.material)
        
    def clear(self):
        self.input_thick.clear()
        self.input_lx.clear()
        self.input_ly.clear()
        self.MplWidget.canvas.axes.clear()
        self.MplWidget.setup_axes()
        self.MplWidget.canvas.draw()
        self.table_R.clearSpans()
        self.model = TableModel(self.data)
        self.table_R.setModel(self.model)
        self.show_fc.setText('-')
        self.show_fd.setText('-')

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()