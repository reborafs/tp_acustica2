import numpy as np
import pandas as pd

#-----------------------------------IMPORT-----------------------------------#

xl = pd.ExcelFile('tabla_materiales.xlsx')
df = xl.parse('Materiales')

print('Los materiales disponibles en la BBDD son:')
print(df.Material)


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

#-----------------------------------EXPORT-----------------------------------#

# df = pd.DataFrame({
#     'Material': ["Madera"],
#     'L1 [m]': [2],
#     'L2 [m]': [2],
#     'Espesor [m]': 0.1,
#     'fc': 180
#     })

# df.to_excel('example.xlsx', index=False)
     
