# column-transformer-in-machine-learning
there is two  way to convert categorical value to numerical value when you have multiple one hot encode and multiple ordinal encoder


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
df=pd.read_csv('/content/archive (7).zip')
df.sample(5)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df.drop(columns=['has_covid']),df['has_covid'],test_size=0.2,random_state=0)
x_train.shape
df.isnull().sum()
Aam zindagi
si = SimpleImputer()
x_train_fever = si.fit_transform(x_train[['fever']])
x_test_fever = si.transform(x_test[['fever']])
x_train_fever.shape
oe=OrdinalEncoder(categories=[['Mild','Strong']])
x_train_cough=oe.fit_transform(x_train[['cough']])
x_test_cough=oe.transform(x_test[['cough']])
x_train_cough.shape
ohe=OneHotEncoder(sparse_output=False,drop='first')
x_train_gender_city=ohe.fit_transform(x_train[['gender','city']])
x_test_gender_city=ohe.fit_transform(x_test[['gender','city']])
x_train_gender_city.shape
x_train_age=x_train.drop(columns=['gender','fever','cough','city']).values
x_test_age=x_test.drop(columns=['gender','fever','cough','city']).values
x_train_age.shape
x_train_transformrd=np.concatenate((x_train_age,x_train_fever,x_train_gender_city,x_train_cough),axis=1)
x_test_transformrd=np.concatenate((x_test_age,x_test_fever,x_test_gender_city,x_test_cough),axis=1)
df_transformed=pd.DataFrame(x_train_transformrd,columns=['age','fever','gender_Female','city_Delhi','city_Kolkata','city_Mumbai','cough'])
display(df_transformed.sample(5))
Mentos zindagi

from sklearn.compose import ColumnTransformer
transformer=ColumnTransformer(transformers=[('tnf1',SimpleImputer(),['fever']),
                                            ('tnf2',OrdinalEncoder(categories=[['Mild','Strong']]),['cough']),
                                            ('tnf3',OneHotEncoder(sparse_output=False,drop='first'),['gender','city'])],remainder='passthrough')

                                          #reminder have two type pass through or drop passthrough ignore another line value
transformer.fit_transform(x_train)


x_train_transformed_df = pd.DataFrame(transformer.fit_transform(x_train), columns=['fever', 'cough', 'gender_Female', 'city_Delhi', 'city_Kolkata', 'city_Mumbai', 'age'])
display(x_train_transformed_df.head())
