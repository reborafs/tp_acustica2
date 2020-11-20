import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# class Material:
#     def __init__(self):
#         self.name = "madera xd"
#         self.thickness = 1
#         self.l2 = 1
#         self.height = 1
#         self.density = 1
#         self.young = 1
#         self.poisson = 1
                
#----------------------------------BANDA-------------------------------------#


bandas = {'octava' : [31.5,63,125,250,500,1000,2000,4000,8000,16000],
          'tercio' : [20,25,31.5,40,50,63,80,100,125,160,
                     200,250,315,400,500,630,800,1000,
                     1250,1600,2000,2500,3150,4000,5000,
                     6300,8000,10000,12500,16000,20000]}

# Ingrese si la frecuencia estara en tercio de octava o octavas: 
tipo_banda = 'tercio'

if(tipo_banda=='octava'):
    freqs = np.array(bandas['octava'])
elif(tipo_banda=='tercio'):
    freqs = np.array(bandas['tercio'])

#----------------------------------EXPORT------------------------------------#

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

materiales = pd.read_excel('tabla_materiales.xlsx')

#-----------------------------------INPUT------------------------------------#

tipo_de_material = 'PYL'
densidad, young, ninterno, poisson = parametro(tipo_de_material)  

c0 = 343                        #Velocidad de propagación en el aire [m/s]
rho0 = 1.18                     #Densidad del aire [kg/m3].
espesor = 0.0125                   #Espesor del material.
l1 = 6                            #Ancho del material
l2 = 4                          #Largo del material

cl = np.sqrt(young/densidad)    #Velocidad de propagación en el material [m/s]
ms = densidad * espesor                         #Masa superficial
B = (young*(espesor**3))/(12*(1-(poisson**2)))  #B
fc = ((c0**2)/(2*np.pi))*(np.sqrt(ms/B))        #Frecuencia critica
k0 = 2*np.pi*freqs/c0                           #Numero de onda
vlambda = np.sqrt(freqs/fc)                     #Lambda

n_tot = ninterno + (ms/(485*np.sqrt(freqs)))

cos21Max = 0.9


#-----------------------------------MODELO-----------------------------------#
#                       Modelo Davy para paneles dobles.        


def single_leaf_davy(freqs, density, young, poisson, espesor, lossfactor, l1, l2):

    ms = densidad * espesor                         #Masa superficial
    B = (young*(espesor**3))/(12*(1-(poisson**2)))  #B
    fc = ((c0**2)/(2*np.pi))*(np.sqrt(ms/B))    
    normal = rho0 * c0 / (np.pi * freqs * ms)
    normal2 = normal * normal
    e = 2*l1*l2/(l1+l2)
    cos2l = c0/(2*np.pi*freqs*e)
    
    cos2l[cos2l>cos21Max] = cos21Max
        
    tau1 = normal2*np.log((normal2 + 1) / (normal2 + cos2l))
    ratio = freqs/fc
    
    r = 1-1/ratio
    r[r<0] = 0
    
    G = np.sqrt(r)
    rad = sigma(G, freqs, l1, l2)
    rad2 = rad * rad
    netatotal = lossfactor + rad * normal
    z = 2 / netatotal
    y = np.arctan(z)-np.arctan(z*(1-ratio))
    tau2 = normal2 * rad2 * y / (netatotal * 2 * ratio)
    tau2 = tau2 * shear(freqs, density, young, poisson, espesor)
    
    tau = np.zeros_like(freqs)
    tau[freqs<fc] = tau1[freqs<fc] + tau2[freqs<fc]
    tau[freqs>=fc] = tau2[freqs>=fc]
    
    single_leaf = -10 * np.log10(tau)

    return single_leaf

def sigma(G, freqs, l2, l1):
    # Definición de constantes:
    c0 = 343
    w = 1.3
    beta = 0.234
    n = 2
    
    S = l1*l2
    U = 2 * (l1 + l2)
    twoa = 4 * S / U
    k = 2 * np.pi * freqs / c0
    f = w * np.sqrt(np.pi/(k*twoa))
    f[f>1] = 1
    
    h = 1 / (np.sqrt(k * twoa / np.pi) * 2 / 3 - beta)
    q = 2 * np.pi / (k * k * S)
    qn = q**n
    
    alpha = h / f - 1
    xn = np.zeros_like(freqs)
    
    xn[G<f] = (h[G<f] - alpha[G<f]*G[G<f])**n
    xn[G>=f] = G[G>=f]**n

    rad = (xn + qn)**(-1 / n)

    return rad

def shear(freqs, densidad, young, poisson, espesor):
    chi = ((1 + poisson) / (0.87 + 1.12*poisson))**2
    X = espesor**2 /12
    QP = young / (1-poisson**2)
    C = -((2*np.pi*freqs)**2)
    B = C*(1 + 2*chi/(1-poisson))*X
    A = X*QP / densidad
    kbcor2 = (np.sqrt(B**2 - 4*A*C) - B) / (2*A)
    kb2 = np.sqrt((-C) / A)
    G = young/(2*(1+poisson))
    kT2 = -C * densidad * chi / G
    kL2 = -C * densidad / QP
    kS2 = kT2 + kL2
    ASI = 1 + X*((kbcor2*kT2 / kL2) - kT2)
    ASI *= ASI
    BSI = 1 - X*kT2 + kbcor2 * kS2 / (kb2**2)
    CSI = np.sqrt(1- X*kT2 + (kS2**2)/(4*kb2*kb2))
    
    return ASI / (BSI*CSI)


R = single_leaf_davy(freqs, densidad, young, poisson, espesor, n_tot, l1,l2)
R = np.round(R,2)
