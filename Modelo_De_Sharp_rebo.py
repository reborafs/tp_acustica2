# -*- c0ding: utf-8 -*-
"""
Created on Mon Nov 16 17:31:32 2020

@author: ACER
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d



# =============================================================================
# Se lee los datos del excel
# =============================================================================

materiales = pd.read_excel('tabla_materiales.xlsx')

tipo_de_material = 'Ladrillo'       #Ingreso material


# =============================================================================
# Tercio de octava o octava
# =============================================================================

bandas = {'octava' : [31.5,63,125,250,500,1000,2000,4000,8000,16000],
          'tercio' : [20,25,31.5,40,50,63,80,100,125,160,
                      200,250,315,400,500,630,800,1000,
                      1250,1600,2000,2500,3150,4000,5000,
                      6300,8000,10000,12500,16000,20000]}

tipos_de_frecuencia = 'tercio'      #'tercio'/'octava'

def octava_terciodeoctava(tipos_de_frecuencia):
    
    if(tipos_de_frecuencia=='octava'):
        freqs = np.array(bandas['octava'])
    elif(tipos_de_frecuencia=='tercio'):
        freqs = np.array(bandas['tercio'])
    else:
        print('Error')
    
    return freqs
    
freqs = octava_terciodeoctava(tipos_de_frecuencia)    

# =============================================================================
#  Se obtienen los parametros del material   
# =============================================================================
def parametro(tipo_de_material):

    a = materiales['Material'] == tipo_de_material
        
    for i in range(len(a)):
        
        if a[i]==True:
      
            datos = materiales.iloc[i]
    
            densidad  = datos.loc['Densidad']
            E = datos.loc['Módulo de Young']
            ninterno = datos.loc['Factor de pérdidas']
            tau = datos.loc['Módulo Poisson']
            

    return densidad,E,ninterno,tau

densidad, E, ninterno, tau = parametro(tipo_de_material)
      

def modelo_de_sharp(E, t, tau, c0, lx, ly, densidad, ninterno, freqs, rho_aire):
    """
    E = Módulo de Young
    t = Espesor
    tau = Módulo de Poisson
    c0 = Velocidad del sonido
    lx = Ancho del material
    ly = Largo del material
    ninterno = Factor de pérdidas
    freqs = Las frecuencias que c0mponen una octava o un tercio de octava
    rho_aire = Densidad del aire
    """                        
# =============================================================================
#       Calculos necesarios
# =============================================================================
   
    ms = densidad * t
    B = (E*(t**3))/(12*(1-(tau**2)))
    fc = ((c0**2)/(2*np.pi))*(np.sqrt(ms/B))
    # fd = (E/(2*np.pi*densidad)) * (np.sqrt(ms/B))
    # f11 = ((c0**2)/(4*fc))*((1/(l1**2))+(1/(l2**2)))


# =============================================================================
#       Aca empieza el metodo   
# =============================================================================
    
    ntotal = ninterno + (ms/(485*np.sqrt(freqs)))
    R1 = 10*np.log10(1+((np.pi*ms*freqs)/(rho_aire*c0))**2) + 10*np.log10((2*ntotal*freqs)/(np.pi*fc))     
    R2 = (10*np.log10(1+((np.pi*ms*freqs)/(rho_aire*c0))**2)) - 5.5
    
    R = np.array(np.zeros(len(freqs)))
    
    # Cuando f<0.5*fc
    R[freqs<fc/2] = R2[freqs<fc/2]
    
    # Cuando f>=fc
    R[freqs>=fc] = np.minimum(R1[freqs>=fc],R2[freqs>=fc])

    # # Cuando 0.5*fc<=f<fc
    # x1 = freqs[freqs<0.5*fc][-1]
    # x2 = freqs[freqs>=fc][0]
    # y1 = R[freqs<0.5*fc][-1]
    # y2 = R[freqs>=fc][0]
    # slope = (y1-y2)/(x1-x2)
    # intercept = (x1*y2 - x2*y1)/(x1-x2)
    # R_fit = slope*freqs + intercept
    # R[(freqs>=0.5*fc)&(freqs<fc)] = R_fit[(freqs>=0.5*fc)&(freqs<fc)]

    # Cuando 0.5*fc<=f<fc
    x1 = freqs[freqs>0.5*fc][0]
    x2 = freqs[freqs>=fc][0]
    y1 = R2[freqs>0.5*fc][0]
    y2 = R[freqs>=fc][0]
    slope = (y1-y2)/(x1-x2)
    intercept = (x1*y2 - x2*y1)/(x1-x2)
    R_fit = slope*freqs + intercept
    R[(freqs>=0.5*fc)&(freqs<fc)] = R_fit[(freqs>=0.5*fc)&(freqs<fc)]
        
    return R,freqs

# t = 0.1
# lx = 4
# ly = 3
# c0 = 343
# rho_aire = 1.18
# R,freqs = modelo_de_sharp(E, t, tau, c0, lx, ly, densidad, 
#                           ninterno, freqs, rho_aire)


# R = np.round(R,2)
# plt.figure()
# plt.semilogx(freqs,R)
# plt.xlim(100,200)
# plt.ylim(0,100)

t = 0.1
lx = 4
ly = 3
c0 = 343
rho_aire = 1.18
ms = densidad * t
B = (E*(t**3))/(12*(1-(tau**2)))
fc = ((c0**2)/(2*np.pi))*(np.sqrt(ms/B))
# fd = (E/(2*np.pi*densidad)) * (np.sqrt(ms/B))
# f11 = ((c0**2)/4*fc)*((1/(lx**2))+(1/(ly**2)))

ntotal = ninterno + (ms/(485*np.sqrt(freqs)))
R1 = 10*np.log10(1+((np.pi*ms*freqs)/(rho_aire*c0))**2)+10*np.log10((2*ntotal*freqs)/(np.pi*fc))     
R2 = (10*np.log10(1+((np.pi*ms*freqs)/(rho_aire*c0))**2)) - 5.5

R = np.array(np.zeros(len(freqs)))

# Cuando f<0.5*fc
R[freqs<fc/2] = R2[freqs<fc/2]

# Cuando f>=fc
R[freqs>=fc] = np.minimum(R1[freqs>=fc],R2[freqs>=fc])

# Cuando 0.5*fc<=f<fc
n_fc = ninterno+(ms/(485*np.sqrt(fc)))
R1_fc = 10*np.log10(1+((np.pi*ms*fc)/(rho_aire*c0))**2)+10*np.log10((2*n_fc*fc)/(np.pi*fc))
R2_fc = (10*np.log10(1+((np.pi*ms*fc)/(rho_aire*c0))**2))-5.5

x1 = fc/2
x2 = fc
y1 = (10*np.log10(1+((np.pi*ms*x1)/(rho_aire*c0))**2))-5.5
y2 = min(R1_fc, R2_fc)
slope = (y1-y2)/(x1-x2)
intercept = (x1*y2 - x2*y1)/(x1-x2)
R_fit = slope*freqs + intercept
R[(freqs>=0.5*fc)&(freqs<fc)] = R_fit[(freqs>=0.5*fc)&(freqs<fc)]
R = np.round(R,2)
