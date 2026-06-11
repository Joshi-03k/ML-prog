import numpy as np
import pandas as pd
from io import StringIO
narr=np.array([[1,2,3,4],[np.nan,6,7,8],[9,10,np.nan,12]])
print(narr)

df=pd.DataFrame(narr,columns=['A','B','C','D'])
print("---1---")
print(df.isnull().sum())
print("---2---")
print(df.values)
print("---3---")
print(df.dropna())
print("---4---")
print(df.dropna(axis=1))
print("---5---")
print(df.dropna(how='all'))
print("---6---")
print(df.dropna(thresh=4))
print("---7---")
print(df.dropna(subset=['A']))
print("---8---")
print(df.dropna(axis=1,how='all'))



