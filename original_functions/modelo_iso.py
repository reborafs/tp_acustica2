import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

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

materiales = pd.read_excel('tabla_materiales.xlsx')



#-------------------------------OCTAVA/TERCIOS-------------------------------#

def octave_thirdoctave(frequency_band):
    '''
    Returns a numpy array, with frequency values ranging 20 Hz to 20Khz.
    If you type 'octave' it will return 10 values, from 31.5 Hz to 16KHz.
    If you type 'third' it will return 10 values, from 31.5 Hz to 16KHz.
    '''
    band_type = {'octave' : [31.5,63,125,250,500,1000,2000,4000,8000,16000],
              'third' : [20,25,31.5,40,50,63,80,100,125,160,
                         200,250,315,400,500,630,800,1000,
                         1250,1600,2000,2500,3150,4000,5000,
                         6300,8000,10000,12500,16000,20000]}
    
    if(frequency_band=='octave'):
        freqs = np.array(band_type['octave'])
    elif(frequency_band=='third'):
        freqs = np.array(band_type['third'])
    else:
        print('Error')
    
    return freqs
    
    
#----------------------------CALCULOS/PARAMETROS-----------------------------#

def iso_model(freqs, young, thickness, poisson, lx, ly, density, ninterno, c0 = 343):

    if lx>ly:
        l1 = lx
        l2 = ly
    else:
        l1 = ly
        l2 = lx
            
    ms = density * thickness                         #Masa superficial
    B = (young*(thickness**3))/(12*(1-(poisson**2)))  #B
    
    fc = ((c0**2)/(2*np.pi))*(np.sqrt(ms/B))        #Frecuencia critica
    k0 = 2*np.pi*freqs/c0                           #Numero de onda
    vlambda = np.sqrt(freqs/fc)                     #Lambda
    
    delta1 = (((1-(vlambda**2))*np.log((1+vlambda)/(1-vlambda))+2*vlambda)/
              (4*(np.pi**2)*((1-(vlambda**2))**1.5)))
    delta2 = np.hstack(((8*(c0**2)*(1-2*(vlambda[freqs<=fc/2]**2)))/
             ((fc**2)*(np.pi**4)*l1*l2*vlambda[freqs<=fc/2]*np.sqrt(1-vlambda[freqs<=fc/2]**2)),
             np.zeros_like(freqs[freqs>fc/2])))
             
    # Cálculo del factor SIGMA de radiación para ondas libres 
    sigma1 = 1/np.sqrt(1-(fc/freqs))
    sigma2 = 4*l1*l2*((freqs/c0)**2)
    sigma3 = np.sqrt((2*np.pi*freqs*(l1+l2))/(16*c0))
    f11 = ((c0**2)/(4*fc))*(l1**-2+l2**-2)          #Frecuencia 1er modo axial
    
    if f11<=fc/2:
        # sigma_d = ((2*(l1+l2))/(l1*l2))*(c0/fc)*delta1[freqs<fc] + delta2[freqs<fc]
        sigma_d = ((2*(l1+l2))/(l1*l2))*(c0/fc)*delta1[freqs<fc]
        sigma2 = sigma2[freqs<fc]
        freqs_under_fc = freqs[freqs<fc]
        sigma_d[(freqs_under_fc<f11)&(sigma_d>sigma2)] = sigma2[(freqs_under_fc<f11)&(sigma_d>sigma2)]
        sigma = np.hstack((sigma_d,sigma1[freqs>=fc]))
        
    elif f11>fc/2:
        if sigma2<sigma3 and sigma1<sigma3:
            sigma = np.hstack((sigma2[freqs<fc],sigma1[freqs>fc]))
        elif sigma2<sigma3 and sigma1>=sigma3:
            sigma = np.hstack((sigma2[freqs<fc],sigma3[freqs>fc]))
        elif sigma2>=sigma3 and sigma1<sigma3:
            sigma = np.hstack((sigma3[freqs<fc],sigma1[freqs>fc]))
        else:
            sigma = sigma3
    
    # Se limit a sigma<=2 segun la norma.
    sigma[sigma>=2] = 2
    
    n_tot = ninterno + (ms/(485*np.sqrt(freqs)))
    pico = -0.964-(0.5+(l2/(l1*np.pi)))*np.log(l2/l1)+(5*l2)/(2*np.pi*l1)-1/(4*np.pi*l1*l2*(k0**2))
    sigma_f = 0.5*(np.log(k0*np.sqrt(l1*l2)) - pico)
    sigma_f[sigma_f>=2] = 2
    sigma_f = np.abs(sigma_f)
    
    
    # 1st condition f < fc
    f1 = freqs[freqs<fc]
    n1_tot = n_tot[freqs<fc]
    sigma1_f = sigma_f[freqs<fc]
    a = (2*rho0*c0)/(2*np.pi*f1*ms)
    b = 2*sigma1_f+(((l1+l2)**2)/(l1**2+l2**2))*np.sqrt(fc/f1)*((sigma[freqs<fc]**2)/n1_tot)
    # b[0] = b[0]*-1
    tau1 = (a**2)*b
    
    # 2nd condition f >= fc
    f2 = freqs[freqs>=fc]
    n2_tot = n_tot[freqs>=fc]
    a2 = (2*rho0*c0)/(2*np.pi*f2*ms)
    b2 = (np.pi*fc*(sigma[freqs>=fc]**2))/(2*f2*n2_tot)
    tau2 = (a2**2)*b2
    
    tau = np.hstack((tau1,tau2))
    R = -10*np.log10(tau)

    return R

#-----------------------------------TEST-------------------------------------#


material = 'Ladrillo'
c0 = 343                        #Velocidad de propagación en el aire [m/s]
rho0 = 1.18                     #Densidad del aire [kg/m3].
thickness = 0.1                 #Espesor del material.
lx = 4                          #Ancho del material
ly = 3                          #Largo del material

densidad, young, ninterno, poisson = parametro(material)  
freqs = octave_thirdoctave('third')    
density, young, ninterno, tau = parametro(material)
R = iso_model(freqs, young, thickness, poisson, lx, ly, density, ninterno)
R = np.round(R,2)

plt.figure()
plt.semilogx(freqs,R)