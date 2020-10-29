import numpy as np

class Material:
    def __init__(self):
        self.name = "madera xd"
        self.thickness = 1
        self.width = 1
        self.height = 1
        self.density = 1
        self.young = 1
        self.poisson = 1
                

# Modelo Davy para paneles dobles.        

def davy_model():
    pass

def filtr():
    pass

def shear(freq, density, young, poisson, thickness):
    chi = ((1 + poisson) / (0.87 + 1.12*poisson))**2
    X = thickness**2 /12
    QP = young / (1-poisson**2)
    C = -((2*np.pi*freq)**2)
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