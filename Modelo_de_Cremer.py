# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 17:37:48 2020

@author: Franco
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# =============================================================================
# Se lee los datos del excel
# =============================================================================

materiales = pd.read_excel('tabla_de_materiales.xlsx',
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
      


def modelo_de_cremer(E, t, tau, co, lx, ly, densidad, ninterno, f):


    """
    E = Módulo de Young
    t = Espesor
    tau = Módulo de Poisson
    co = Velocidad del sonido
    lx = Ancho del material
    ly = Largo del material
    ninterno = Factor de pérdidas
    f = Las frecuencias que componen una octava o un tercio de octava
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
    
    ntotal = np.array(np.zeros(len(f)))
    
    for i in range(len(f)):  
        
        if(f[i]<fc):
            
            R[i] = 20*np.log10(ms*f[i])-47
            
        elif(fd > f[i] > fc):
            
            ntotal[i] = ninterno + (ms/(485*np.sqrt(f[i])))
            R[i] = 20*np.log10(ms*f[i]) - 10*np.log10(np.pi/(4*ntotal[i])) - 10*np.log10(fc/(f[i]-fc)) - 47    
            
        else:
            
            R[i] = 20*np.log10(ms*f[i])-47
            
    return R, f
    
    
R,f = modelo_de_cremer(E,0.1, tau, 343, 4, 3, densidad, ninterno, f)   
        
plt.figure()
plt.semilogx(f,R)