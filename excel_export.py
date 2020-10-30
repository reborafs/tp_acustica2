import numpy as np
import pandas as pd

df = pd.DataFrame({
    'Material': ["Madera"],
    'L1 [m]': [2],
    'L2 [m]': [2],
    'Espesor [m]': 0.1,
    'fc': 180
    })

df.to_excel('example.xlsx', index=False)
     
