import numpy as np
import pandas as pd

#-----------------------------------IMPORT-----------------------------------#

def import_material(material_name):
    '''
    Input a string with the desired material and returns a dict with its 
    density, Young's modulus, loss factor and Poisson's modulus.
    The data will be imported from an excel file called 'tabla_materiales.xlsx'
    '''
    xl = pd.ExcelFile('tabla_materiales.xlsx')
    df = xl.parse('Materiales')   
    a = df['Material'] == material_name
        
    for i in range(len(a)):
        
        if a[i]==True:
    
            datos = df.iloc[i]
            
            parameters = {'name':material_name,
              'density': datos.loc['Densidad'],
              'young': datos.loc['Módulo de Young'],
              'poisson': datos.loc['Módulo Poisson'],
              'loss factor': datos.loc['Factor de pérdidas']}
    
    return parameters

#-----------------------------------EXPORT-----------------------------------#

def export_to_excel():
    df = pd.DataFrame({
        'Material': ["Madera"],
        'L1 [m]': [2],
        'L2 [m]': [2],
        'Espesor [m]': 0.1,
        'fc': 180
        })
    
    df.to_excel('export.xlsx', index=False)
     
