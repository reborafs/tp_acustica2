import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

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

c0 = 343                        #Velocidad de propagación en el aire [m/s]
rho0 = 1.18                   #Densidad del aire [kg/m3].
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


# Array de frecuencia critica
fp = cl/(5.5*espesor)           #Frecuencia de placa
fc = np.ones_like(freqs)*((c0**2)/(1.8*cl*espesor))   #Frecuencia critica
fc[(freqs>fc)&(freqs<fp)] = (
    fc[(freqs>fc)&(freqs<fp)]*(((4.05*espesor*freqs[(freqs>fc)&(freqs<fp)])/cl)+
        np.sqrt(1+(4.05*espesor*freqs[(freqs>fc)&(freqs<fp)]/cl))))
fc[(freqs>fc)&(freqs>=fp)] = 2*fc[(freqs>fc)&(freqs>=fp)]*((freqs[(freqs>fc)&(freqs>=fp)]/fc[(freqs>fc)&(freqs>=fp)])**3)
k0 = 2*np.pi*freqs/c0           #Numero de onda
vlambda = np.sqrt(freqs/fc)     #Lambda

delta1 = (((1-(vlambda**2))*np.log((1+vlambda)/(1-vlambda))+2*vlambda)/
          (4*(np.pi**2)*((1-(vlambda**2))**1.5)))
delta2 = np.hstack(
        ((8*(c0**2)*(1-2*(vlambda[freqs<fc/2]**2)))/
         ((fc[freqs<fc/2]**2)*(np.pi**4)*l1*l2*vlambda[freqs<fc[freqs<fc/2]/2]*np.sqrt(1-vlambda[freqs<fc/2]**2)),
         np.zeros_like(freqs[freqs>=fc[freqs<fc/2]/2])))
         

# Cálculo del factor SIGMA de radiación para ondas libres 

sigma1 = 1/np.sqrt(1-(fc/freqs))
sigma2 = 4*l1*l2*((freqs/c0)**2)
sigma3 = (2*np.pi*freqs*(l1+l2))/np.sqrt(16*c0)
f11 = ((c0**2)/(4*fc))*(l1**-2+l2**-2)      #Frecuencia 1er modo axial

if f11<=fc/2:
    sigma = ((2*(l1+l2)*c0*delta1[freqs<fc])/(l1*l2*fc))+delta2[freqs<fc]
    sigma2 = sigma2[freqs<fc]
    sigma[(freqs[freqs<fc]<f11) & (sigma>sigma2)] = sigma2[(freqs[freqs<fc]<f11) & (sigma>sigma2)]
    sigma1 = sigma1[freqs>=fc]
    sigma = np.hstack((sigma,sigma1))
    
elif f11>fc/2:
    if sigma2<sigma3 and sigma1<sigma3:
        sigma = np.hstack((sigma2[freqs<fc],sigma1[freqs>fc]))
    elif sigma2<sigma3 and sigma1>=sigma3:
        sigma = np.hstack((sigma2[freqs<fc],sigma3[freqs>fc]))
    elif sigma2>=sigma3 and sigma1<sigma3:
        sigma = np.hstack((sigma3[freqs<fc],sigma1[freqs>fc]))
    else:
        sigma = sigma3

masa_s = densidad*espesor
# n_tot = ninterno + (2*rho0*c0*sigma)/(2*np.pi*freqs*masa_s) + (c0/((np.pi**2))
n_tot = ninterno + masa_s/(485*np.sqrt(freqs))
pico = -0.9564-(0.5+(l2/(l1*np.pi)))*np.log(l2/l1)+5*l2/(2*np.pi*l1)-1/(4*np.pi*l1*l2*(k0**2))
sigma_f = 0.5*(np.log(k0*np.sqrt(l1*l2))- pico)


# 1st condition f < fc
f1 = freqs[freqs<fc]
n1_tot = n_tot[freqs<fc]
sigma1_f = sigma_f[freqs<fc] 
a = (2*rho0*c0)/(2*np.pi*f1*masa_s)
b = 2*sigma1_f+(((l1+l2)**2)/(l1**2+l2**2))*np.sqrt(fc/f1)*((sigma[freqs<fc]**2)/n1_tot)
tau1 = (a**2)*b

# 2nd condition f >= fc
f2 = freqs[freqs>=fc]
n2_tot = n_tot[freqs>=fc]
a = (2*rho0*c0)/(2*np.pi*f2*masa_s)
b = (np.pi*fc*(sigma[freqs>=fc]**2))/(2*f2*n2_tot)
tau2 = (a**2)*b

tau = np.hstack((tau1,tau2))
R = -10*np.log10(tau)
R = np.round(R,2)
plt.figure()
plt.semilogx(freqs,R)
plt.ylim(0,100)
plt.xlim(0,22000)
# plt.xticks(freqs, freqs, rotation=70)
plt.grid()

