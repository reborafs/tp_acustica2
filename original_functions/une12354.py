import numpy as np 

def fc (cl, t, co= '1.18'):
    ''' 
    fc= Frecuencia crítica del material
    
    Entradas:
    #co = velocidad de propagación de la onda en el aire [m/s]
    #cl = velocidad de propagación de la onda en el material [m/s]
    #t = espesor del material [m] '''
    
    fc= co**2/(1.8*cl*t)
    
    return fc


def ko(f, co= '1.18'):
    '''
    ko= número de onda
    
    Entradas: 
    #co = velocidad de propagación de la onda en el aire [m/s]
    #f = frecuencia [Hz]
    ''' 
    ko= 2*np.pi*f/co
    
    return ko



def f11(l1, l2, fc, co= '1.18'):
    '''
    f11= frecuencia de comparación para SIGMA (factor de radiación para ondas 
    de flexión libres)
    
    Entradas:

    l1= longitud del material
    l2= altura del material
    fc= frecuencia crítica
    co= velocidad de propagación de la onda en el aire
    
    '''
    
    fc=fc(cl, t)
    
    f11=(co**2/4*fc)*(l1**-2 + l2**-2)
    
    return f11



def f_lambda (f, fc):
    '''
    flamda= Lambda, para calcular la densidad d1 y d2
    
    Entradas:
    f= frecuencia
    fc= frecuencia crítica 
    
    '''
    flambda = np.sqrt(f/fc)
    
    return flambda


def sigma_f(ko, l1, l2):
     
    pico = -0.9564- (0.5+ (l2/(l1*np.pi)))*np.ln(l2/l1) + (5*l2/ 2*np.pi*l1)- (1/ 4*np.p1*l1*l2*(ko**2))
    sigma_f = 0.5* (np.ln(ko*np.sqrt(l1*l2))- pico)
    
    return sigma_f

def sigma(f, fc, f11, l1, l2, co= '1.18'):
  ''' 
  cálculo del factor de radiación para ondas libres 
  '''
  sigma1= (1-(fc/f))**(-1/2)
  sigma2 = 4*l1*l2*((f/co)**2)  
  sigma3 = (2*np.pi*f*(l1+l2))/((16*co)**(1/2))
  sigma4 = ((2*(l1+l2)*co*d1)/(l1*l2*fc))+d2
  
  if f11<=(fc/2):
      if fc<=f: 
          sigma = sigma1
      
      else:
          vlambda= f_lambda(f, fc)
          d1= ((1-vlambda**2)*np.ln((1+vlambda)/(1-vlambda))+
               2*vlambda)/(4*(3.1415**2)*(1-vlambda**2)**(1.5))
          if f>fc/2:
              d2= 0
          else:
             d2= ((8*(co**2)*(1-2*(vlambda)**2)))/(fc**2*(3.1415)**4*l1*l2*vlambda*(1-vlambda**2)**(1/2))
        

      

def tau(cl, t, fc, l1, l2):
    '''
    tau= factor de transmisión
     
    Entradas: 
    
    #cl = velocidad de propagación de la onda en el material [m/s]
    #t = espesor del material [m]
    #fc = frecuencia crítica [Hz]
    #l1 = longitud del material 
    #l2 = longitud del material
    '''

    #ms= masa superficial
    #fc frecuencia crítica
    #nt= factor de pérdida total
    #sigma= factor de radiación para ondas de flexión libres
    #sigma_f = factor de radiación para 
    #l1, l2 = longitudes de los bordes rectangulares 
    #ko= número de onda
    #ro= densidad del aire
    #co= velocidad de propagación en el aire
    
    f= [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000]
     
    for i in f:
        ko = ko(f[i])
        if i >= fc:
            sigma = (1/np.sqrt(1-(fc/f)))
            tau= ((2*ro*co)/(2*np.pi*f*ms))**2*((np.pi*fc*(sigma**2))/(2*f*nt))
        else:
            sigma = 1
            tau= ((2*ro*co)/(2*np.pi*f*ms))**2*(2*sigma_f+((l1 + l2)**2)/(l1**2 + l2 **2 ))*(np.sqrt(fc/f))*((sigma**2)/nt)  
    return tau
                   