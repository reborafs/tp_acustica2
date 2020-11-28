import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from import_export import import_material
from model_process import cremer_model, sharp_model, iso_model, davy_model, shear, sigma, octave_thirdoctave

class Material():
    
    def __init__(self, material_dict):
        self.name = material_dict['name']
        self.density = material_dict['density']
        self.loss_factor = material_dict['loss factor']
        self.poisson = material_dict['poisson']
        self.young = material_dict['young']

    def freq_c(self, lx, ly, thickness, c0=343):
        ms = self.density * thickness
        B = (self.young*(thickness**3))/(12*(1-(self.poisson**2)))
        fc = ((c0**2)/(2*np.pi))*(np.sqrt(ms/B))
        return fc
        
    def freq_d(self, lx, ly, thickness):
        ms = self.density * thickness
        B = (self.young*(thickness**3))/(12*(1-(self.poisson**2)))
        fd = (self.young/(2*np.pi*self.density)) * (np.sqrt(ms/B))
        return fd
    
    def cremer(self, lx, ly, thickness, band_type='third'):
        freqs = octave_thirdoctave(band_type)
        R_cremer = cremer_model(freqs, lx, ly, thickness, self.young, 
                                self.poisson, self.density, self.loss_factor)
        return R_cremer
        
    def sharp(self, lx, ly, thickness, band_type='third'):
        freqs = octave_thirdoctave(band_type)
        R_sharp = sharp_model(freqs, lx, ly, thickness, self.young, 
                              self.poisson, self.density, self.loss_factor)
        return R_sharp
    
    def iso12354(self, lx, ly, thickness, band_type='third'):
        freqs = octave_thirdoctave(band_type)
        R_iso = iso_model(freqs, lx, ly, thickness, self.young, 
                          self.poisson, self.density, self.loss_factor)
        return R_iso
    
    def davy(self, lx, ly, thickness, band_type='third'):
        freqs = octave_thirdoctave(band_type)
        R_davy = davy_model(freqs, lx, ly, thickness, self.young, 
                            self.poisson, self.density, self.loss_factor)
        return R_davy
        
#------------------------------------TEST------------------------------------#

# material_name = 'Ladrillo'
# material = Material(import_material(material_name))
# thickness = 0.1
# lx = 4
# ly = 3

# band_type = 'third'
# freqs = octave_thirdoctave(band_type)

# R_cremer = material.cremer(lx,ly,thickness, band_type)
# R_sharp = material.davy(lx,ly,thickness)
# R_iso = material.iso12354(lx,ly,thickness)
# R_davy = material.davy(lx,ly,thickness)

# # Plotting
# plt.figure()
# plt.title('Single leaf wall insulation')
# plt.semilogx(freqs, R_cremer)
# plt.semilogx(freqs, R_sharp)
# plt.semilogx(freqs, R_iso)
# plt.semilogx(freqs, R_davy)
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('R [dB]')
# plt.xticks((20,100,1000,10000,20000),(20,100,'1k','10k','20k'))
# plt.grid(which='both')

# plt.legend(('Cremer', 'Sharp', 'ISO 12354-1', 'Davy'))