import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd



# f = 200                     #Frecuencia

# ko= 2*np.pi*f/co            #Numero de onda

# l1 = 1                      #Longitud del material
# l2 = 2                      #Altura del material

# f11 = (co**2/4*fc)*(l1**-2 + l2**-2) #frecuencia de comparación para SIGMA

# f_lambda = np.sqrt(f/fc)     #Lambda para calcular la densidad d1 y d2


# def sigma_f(ko, l1, l2):
     
    # pico = -0.9564- (0.5+ (l2/(l1*np.pi)))*np.ln(l2/l1) + (5*l2/ 2*np.pi*l1)- (1/ 4*np.p1*l1*l2*(ko**2))
    # sigma_f = 0.5* (np.ln(ko*np.sqrt(l1*l2))- pico)
    
#     return sigma_f


# d1 = 1                      #
# d2 = 2                      #

# cálculo del factor de radiación para ondas libres 
# sigma1= (1-(fc/f))**(-1/2)
# sigma2 = 4*l1*l2*((f/co)**2)  
# sigma3 = (2*np.pi*f*(l1+l2))/((16*co)**(1/2))
# sigma4 = ((2*(l1+l2)*co*d1)/(l1*l2*fc))+d2

# if f11<=(fc/2):
#     if fc<=f: 
#         sigma = sigma1
    
#     else:
#         vlambda = f_lambda
#         d1= ((1-vlambda**2)*np.ln((1+vlambda)/(1-vlambda))+
#              2*vlambda)/(4*(3.1415**2)*(1-vlambda**2)**(1.5))
#         if f>fc/2:
#             d2= 0
#         else:
#            d2= ((8*(co**2)*(1-2*(vlambda)**2)))/(fc**2*(3.1415)**4*l1*l2*vlambda*(1-vlambda**2)**(1/2))
        

      

# def tau(cl, t, fc, l1, l2):
#     '''
#     tau= factor de transmisión
     
#     Entradas: 
    
#     #cl = velocidad de propagación de la onda en el material [m/s]
#     #t = espesor del material [m]
#     #fc = frecuencia crítica [Hz]
#     #l1 = longitud del material 
#     #l2 = longitud del material
#     '''

    #ms= masa superficial
    #fc frecuencia crítica
    #nt= factor de pérdida total
    #sigma= factor de radiación para ondas de flexión libres
    #sigma_f = factor de radiación para 
    #l1, l2 = longitudes de los bordes rectangulares 
    #ko= número de onda
    #ro= densidad del aire
    #co= velocidad de propagación en el aire
    
    # f= [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000]
     
    # for i in f:
    #     ko = ko(f[i])
    #     if i >= fc:
    #         sigma = (1/np.sqrt(1-(fc/f)))
    #         tau= ((2*ro*co)/(2*np.pi*f*ms))**2*((np.pi*fc*(sigma**2))/(2*f*nt))
    #     else:
    #         sigma = 1
    #         tau= ((2*ro*co)/(2*np.pi*f*ms))**2*(2*sigma_f+((l1 + l2)**2)/(l1**2 + l2 **2 ))*(np.sqrt(fc/f))*((sigma**2)/nt)  
    # return tau


# =============================================================================
# NEW CODE
# =============================================================================

def iso_model(E, t, tau, c0, lx, ly, densidad, ninterno, f):
    pass

materiales = pd.read_excel('tabla_materiales.xlsx')
tipo_de_material = 'Ladrillo'

#-----------------------------------INPUT------------------------------------#

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

densidad, young, ninterno, poisson = parametro(tipo_de_material)  

c0 = 340                        #Velocidad de propagación en el aire [m/s]
espesor = 0.1                   #Espesor del material.
l1 = 4                          #Ancho del material
l2 = 3                          #Largo del material

#-------------------------------OCTAVA/TERCIOS-------------------------------#

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
    
#----------------------------CALCULOS/PARAMETROS-----------------------------#

cl = np.sqrt(young/densidad)    #Velocidad de propagación en el material [m/s]
fc= (c0**2)/(1.8*cl*espesor)    #Frecuencia critica
ko= 2*np.pi*freqs/c0            #Numero de onda

sigma = 1
sigma_f = 1
masa_s = densidad*espesor
n_tot = ninterno + (masa_s/(485*np.sqrt(freqs)))

pico = -0.9564- (0.5+ (l2/(l1*np.pi)))*np.log(l2/l1) + (5*l2/ 2*np.pi*l1)- (1/ 4*np.pi*l1*l2*(ko**2))
sigma_f = 0.5* (np.log(ko*np.sqrt(l1*l2))- pico)
sigma = (1/np.sqrt(1-(fc/freqs)))


# 1st condition f <= fc
f1 = freqs[freqs<=fc]
n1_tot = n_tot[freqs<=fc]
sigma1 = 1
a = (2*densidad*c0)/(2*np.pi*f1*masa_s)
b = (np.pi*fc*(sigma1**2))/(2*f1*n1_tot)
tau1 = (a**2)*b


# 2nd condition f <= fc
f2 = freqs[freqs>fc]
n2_tot = n_tot[freqs>fc]
sigma2_f = sigma_f[freqs>fc] 
sigma2 = sigma[freqs>fc]
a = (2*densidad*c0)/(2*np.pi*f2*masa_s)
b = 2*sigma2_f+(((l1+l2)**2)/(l1**2+l2**2))*np.sqrt(fc/f2)*((sigma2**2)/n2_tot)
tau2 = (a**2)*b

tau = np.hstack((tau1,tau2))
R = -10*np.log(tau)


