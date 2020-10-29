import numpy as np
import pandas as pd

df = pd.DataFrame({
    'Col A': [1,2,3],
    'Col B': ['bokita','elmas','grande'],
    'Col C': [np.nan,np.nan,'not empty']
    })

df.to_excel('example.xlsx', index=False)
     
