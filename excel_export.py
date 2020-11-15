import numpy as np
import pandas as pd

#-----------------------------------IMPORT-----------------------------------#

xl = pd.ExcelFile('tabla_materiales.xlsx')
df = xl.parse('Materiales')

print('Los materiales disponibles en la BBDD son:')
print(df.Material)


#-----------------------------------EXPORT-----------------------------------#

# df = pd.DataFrame({
#     'Material': ["Madera"],
#     'L1 [m]': [2],
#     'L2 [m]': [2],
#     'Espesor [m]': 0.1,
#     'fc': 180
#     })

# df.to_excel('example.xlsx', index=False)
     
