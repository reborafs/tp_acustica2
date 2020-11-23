import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Tercio de octava o octava
# =============================================================================

def octave_thirdoctave(frequency_band):
    '''
    Returns a numpy array, with frequency values ranging 20 Hz to 20Khz.
    If you type 'octave' it will return 10 values, from 31.5 Hz to 16KHz.
    If you type 'third' it will return 31 values, from 31.5 Hz to 16KHz.

    Parameters
    ----------
    frequency_band : str
        Type 'octave' or 'third' to choose frequency band type.

    Returns
    -------
    freqs : ndarray
        The desired numpy array.

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
    
# =============================================================================
#  Se obtienen los parametros del material   
# =============================================================================
def parametro(tipo_de_material):

    materiales = pd.read_excel('tabla_materiales.xlsx')    
    a = materiales['Material'] == tipo_de_material
        
    for i in range(len(a)):
        
        if a[i]==True:
    
            datos = materiales.iloc[i]
    
            density  = datos.loc['Densidad']
            E = datos.loc['Módulo de Young']
            ninterno = datos.loc['Factor de pérdidas']
            tau = datos.loc['Módulo Poisson']
            

    return density,E,ninterno,tau

def freq_c(lx, ly, thickness, young, poisson, density, c0=343):
    ms = density * thickness
    B = (young*(thickness**3))/(12*(1-(poisson**2)))
    fc = ((c0**2)/(2*np.pi))*(np.sqrt(ms/B))
    return fc

def cremer_model(freqs, lx, ly, thickness, young, poisson, density, ninterno, c0 = 343):
    """

    """                        
    ms = density * thickness
    B = (young*(thickness**3))/(12*(1-(poisson**2)))
    fc = ((c0**2)/(2*np.pi))*(np.sqrt(ms/B))
    fd = (young/(2*np.pi*density)) * (np.sqrt(ms/B))


    R = np.zeros_like(freqs)
    
    ntotal = np.array(np.zeros(len(freqs)))
    
        
    # If freqs < fc: 
    R[freqs<fc] = 20*np.log10(ms*freqs[freqs<fc])-47
        
    # If fd > freqs[i] > fc:
    condition = (freqs>fc) & (fd>freqs)
    ntotal[condition] = ninterno + (ms/(485*np.sqrt(freqs[condition])))
    R[condition] = (20*np.log10(ms*freqs[condition]) 
                    - 10*np.log10(np.pi/(4*ntotal[condition])) 
                    - 10*np.log10(fc/(freqs[condition]-fc)) - 47)    
    
    # If freqs > fd:
    R[freqs>fd] = 20*np.log10(ms*freqs[freqs>fd]) - 47
            
    return R
    
def sharp_model(freqs, lx, ly, thickness, young, poisson, density, ninterno, rho_aire = 1.18, c0 = 343):
    """

    """                        
    ms = density * thickness
    B = (young*(thickness**3))/(12*(1-(poisson**2)))
    fc = ((c0**2)/(2*np.pi))*(np.sqrt(ms/B))

    ntotal = ninterno + (ms/(485*np.sqrt(freqs)))
    R1 = 10*np.log10(1+((np.pi*ms*freqs)/(rho_aire*c0))**2) + 10*np.log10((2*ntotal*freqs)/(np.pi*fc))     
    R2 = (10*np.log10(1+((np.pi*ms*freqs)/(rho_aire*c0))**2)) - 5.5
    
    R = np.array(np.zeros(len(freqs)))
    
    # IF f<0.5*fc
    R[freqs<fc/2] = R2[freqs<fc/2]
    
    # IF f>=fc
    R[freqs>=fc] = np.minimum(R1[freqs>=fc],R2[freqs>=fc])

    # IF 0.5*fc<=f<fc
    x1 = freqs[freqs>0.5*fc][0]
    x2 = freqs[freqs>=fc][0]
    y1 = R2[freqs>0.5*fc][0]
    y2 = R[freqs>=fc][0]
    slope = (y1-y2)/(x1-x2)
    intercept = (x1*y2 - x2*y1)/(x1-x2)
    R_fit = slope*freqs + intercept
    R[(freqs>=0.5*fc)&(freqs<fc)] = R_fit[(freqs>=0.5*fc)&(freqs<fc)]
        
    return R


def iso_model(freqs, lx, ly, thickness, young, poisson, density, ninterno, c0 = 343, rho0 = 1.18):

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


def davy_model(freqs, lx, ly, thickness, young, poisson, density, lossfactor, c0 = 343, rho0 = 1.18):

    ms = density * thickness                          #Masa superficial
    B = (young*(thickness**3))/(12*(1-(poisson**2)))  #B
    fc = ((c0**2)/(2*np.pi))*(np.sqrt(ms/B))    
    
    normal = rho0 * c0 / (np.pi * freqs * ms)
    normal2 = normal * normal
    e = 2*lx*ly/(lx+ly)
    cos2l = c0/(2*np.pi*freqs*e)
    cos21Max = 0.9
    cos2l[cos2l>cos21Max] = cos21Max
        
    tau1 = normal2*np.log((normal2 + 1) / (normal2 + cos2l))
    ratio = freqs/fc
    
    r = 1-1/ratio
    r[r<0] = 0
    
    G = np.sqrt(r)
    rad = sigma(G, freqs, lx, ly)
    rad2 = rad * rad
    netatotal = lossfactor + rad * normal
    z = 2 / netatotal
    y = np.arctan(z)-np.arctan(z*(1-ratio))
    tau2 = normal2 * rad2 * y / (netatotal * 2 * ratio)
    tau2 = tau2 * shear(freqs, density, young, poisson, thickness)
    
    tau = np.zeros_like(freqs)
    tau[freqs<fc] = tau1[freqs<fc] + tau2[freqs<fc]
    tau[freqs>=fc] = tau2[freqs>=fc]
    
    R = -10 * np.log10(tau)

    return R

def sigma(G, freqs, lx, ly):
    # Definición de constantes:
    c0 = 343
    w = 1.3
    beta = 0.234
    n = 2
    
    S = lx*ly
    U = 2 * (lx + ly)
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

def shear(freqs, density, young, poisson, thickness):
    chi = ((1 + poisson) / (0.87 + 1.12*poisson))**2
    X = thickness**2 /12
    QP = young / (1-poisson**2)
    C = -((2*np.pi*freqs)**2)
    B = C*(1 + 2*chi/(1-poisson))*X
    A = X*QP / density
    kbcor2 = (np.sqrt(B**2 - 4*A*C) - B) / (2*A)
    kb2 = np.sqrt((-C) / A)
    G = young/(2*(1+poisson))
    kT2 = -C * density * chi / G
    kL2 = -C * density / QP
    kS2 = kT2 + kL2
    ASI = 1 + X*((kbcor2*kT2 / kL2) - kT2)
    ASI *= ASI
    BSI = 1 - X*kT2 + kbcor2 * kS2 / (kb2**2)
    CSI = np.sqrt(1- X*kT2 + (kS2**2)/(4*kb2*kb2))
    
    return ASI / (BSI*CSI)


# =============================================================================
#  TESTING MODELS
# =============================================================================

# material = 'Ladrillo'
# thickness = 0.1
# lx = 4
# ly = 3

# freqs = octave_thirdoctave('third')    
# density, young, ninterno, poisson = parametro(material)

# R_cremer = cremer_model(freqs, lx, ly, thickness, young, poisson, density, ninterno)
# R_cremer = np.round(R_cremer,2)

# R_sharp = sharp_model(freqs, lx, ly, thickness, young, poisson, density, ninterno)
# R_sharp = np.round(R_sharp,2)

# R_iso = iso_model(freqs, lx, ly, thickness, young, poisson, density, ninterno)
# R_iso = np.round(R_iso,2)

# R_davy = davy_model(freqs, lx, ly, thickness, young, poisson, density, ninterno)
# R_davy = np.round(R_davy, 2)


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
