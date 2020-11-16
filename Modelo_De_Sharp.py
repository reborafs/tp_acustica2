# -*- coding: utf-8 -*-
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

materiales = pd.read_excel(r'C:/Users/ACER/Documents/Codigos/Python/Tabla_de_materiales.xlsx',
                         skiprows=1, usecols = [1,2,3,4,5,6] )

tipo_de_material = input('Ingrese el tipo de material: ')


# =============================================================================
# Tercio de octava o octava
# =============================================================================

bandas = {'octava' : [31.5,63,125,250,500,1000,2000,4000,8000,16000],
                      'tercio de octava' : [20,25,31.5,40,50,63,80,100,125,160,
                                            200,250,315,400,500,630,800,1000,
                                            1250,1600,2000,2500,3150,4000,5000,
                                            6300,8000,10000,12500,16000,20000]}

tipos_de_frecuencia = input('Ingrese si la frecuencia estara en tercio de octava o octavas: ')


def octava_terciodeoctava(tipos_de_frecuencia):
    
    if(tipos_de_frecuencia=='octava'):
        f = bandas['octava']
    elif(tipos_de_frecuencia=='tercio de octava'):
        f = bandas['tercio de octava']
    else:
        print('Error')
    
    return f
    
f = octava_terciodeoctava(tipos_de_frecuencia)    

# =============================================================================
#  Se obtienen los parametros del material   
# =============================================================================
def parametro(tipo_de_material):

    a = materiales['Material'] == tipo_de_material
        
    for i in range(len(a)):
        
        if a[i]==True:
    
            fila = i
    
            datos = materiales.iloc[i]
    
            densidad  = datos.loc['Densidad']
            E = datos.loc['Módulo de Young']
            ninterno = datos.loc['Factor de pérdidas']
            tau = datos.loc['Módulo Poisson']
            

    return densidad,E,ninterno,tau

densidad, E, ninterno, tau = parametro(tipo_de_material)
      

def modelo_de_sharp(E, t, tau, co, lx, ly, densidad, ninterno, f,ro_del_aire):


    """
    E = Módulo de Young
    t = Espesor
    tau = Módulo de Poisson
    co = Velocidad del sonido
    lx = Ancho del material
    ly = Largo del material
    ninterno = Factor de pérdidas
    f = Las frecuencias que componen una octava o un tercio de octava
    ro_del_aire = Densidad del aire
    """                        
# =============================================================================
# Calculos necesarios
# =============================================================================
   
    ms = densidad * t
    
    B = (E*(t**3))/(12*(1-(tau**2)))
    
    fc = ((co**2)/(2*np.pi))*(np.sqrt(ms/B))
    
    fd = (E/(2*np.pi*densidad)) * (np.sqrt(ms/B))
    
    f11 = ((co**2)/4*fc)*((1/(lx**2))+(1/(ly**2)))

# =============================================================================
#     Aca empieza el metodo   
# =============================================================================
    
    R = np.array(np.zeros(len(f)))
    
    R1 = np.array(np.zeros(len(f)))
    
    R2 = np.array(np.zeros(len(f)))
    
    ntotal = np.array(np.zeros(len(f)))
    
    for i in range(len(f)):
       
        if(f[i]<0.5*fc):
            
            R[i] = (10*np.log10(1+((np.pi*ms*f[i])/(ro_del_aire*co))**2)) - 5.5
        
        elif(f[i]>=fc):
            
            ntotal[i] = ninterno + (ms/(485*np.sqrt(f[i])))
            
            R1[i] = 10*np.log10(1+((np.pi*ms*f[i])/(ro_del_aire*co))**2) + 10*np.log10(1+((2*ntotal[i]*f[i])/(np.pi*fc))) 
            
            R2[i] = (10*np.log10(1+((np.pi*ms*f[i])/(ro_del_aire*co))**2)) - 5.5
            
            if(R1[i]<R2[i]):
                
                R[i] = R1[i]
            
            elif(R1[i]>R2[i]):
                
                R[i] = R2[i]

            else:
                
                R[i] = R1[i]
            
        else:
        
            ntotal[i] = ninterno + (ms/(485*np.sqrt(f[i])))
            
            R1[i] = 10*np.log10(1+((np.pi*ms*f[i])/(ro_del_aire*co))**2) + 10*np.log10(1+((2*ntotal[i]*f[i])/(np.pi*fc))) 
            
            R2[i] = (10*np.log10(1+((np.pi*ms*f[i])/(ro_del_aire*co))**2)) - 5.5
            
            
            
            
    return R,f



R,f = modelo_de_sharp(E,0.1, tau, 343, 4, 3, densidad, ninterno, f,1.18) 
            