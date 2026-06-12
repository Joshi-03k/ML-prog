import numpy as np
import pandas as pd
from io import StringIO
from sklearn.impute import SimpleImputer

narr=np.array([[1,2,3,4],[np.nan,6,7,8],[9,10,np.nan,12]])
print(narr)

df=pd.DataFrame(narr,columns=['A','B','C','D'])

print('-----inputed data with mean-----')
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer=imputer.fit(df)

imputed_data=imputer.transform(df)
print(imputed_data)

print('----imputed data with median-----')
imputer = SimpleImputer(missing_values=np.nan,strategy='median')
imputer=imputer.fit(df)
imputed_data=imputer.transform(df)
print(imputed_data)

print('-----imputed data with constant-----')
imputer = SimpleImputer(missing_values=np.nan,strategy='constant',fill_values)
imputer = imputer.fit(df)
imputed_data=imputer.transfrom(dF)
print(imputed_data)
