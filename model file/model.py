import pandas as pd
import numpy as np

from sklearn.ensemble import ExtraTreesRegressor

df=pd.read_csv('Car_Price_Prediction/car data.csv')

df.drop('Car_Name',axis=1,inplace=True)

df['Current_Year']=2020

df['No_Years']=df['Current_Year']-df['Year']

df.drop(['Year','Current_Year'],axis=1,inplace=True)

df=pd.get_dummies(df,drop_first=True)

x=df.iloc[:,1:]
y=df.iloc[:,0]

regmodel=ExtraTreesRegressor()

regmodel.fit(x,y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.ensemble import RandomForestRegressor

random=RandomForestRegressor()

n_estimators=[int(x) for x in np.linspace(start = 100, stop= 1200,num = 12)]

max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

from sklearn.model_selection import RandomizedSearchCV

random = RandomizedSearchCV(estimator = random, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)

random.fit(x_train,y_train)

predictions=random.predict(x_test)

import pickle

file = open('car_price_prediction.pkl', 'wb')

pickle.dump(random, file)
