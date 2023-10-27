import pandas as pd


data=pd.read_csv("E:train-data.csv")
data.shape

data.head()


data.describe()


data.drop('Name',axis=1,inplace=True)

data.head()

data.drop(['Location'],axis=1,inplace=True)

data.isnull().sum()

data.dropna(inplace=True)



data.head()

data.shape

data.drop('Unnamed: 0',axis=1,inplace=True)

data.shape

data.isnull().sum()



data.shape

data['current_year']=2023

data['no_of_years']=data['current_year']-data['Year']

data.head()

data.drop(['current_year','Year'],axis=1,inplace=True)

data.head()

data.shape

data.info()

data = data.reset_index(drop=True)





for i in range(data.shape[0]):
        data.at[i, 'mileage'] = data['Mileage'][i].split()[0]
        data.at[i, 'engine'] = data['Engine'][i].split()[0]
        data.at[i, 'power'] = data['Power'][i].split()[0]
        data.at[i,'presentprice']= data['New_Price'][i].split()[0]


data.info()

data['mileage'] = data['mileage'].astype(float)
data['engine'] = data['engine'].astype(float)
data['power'] = data['power'].astype(float)
data['presentprice']=data['presentprice'].astype(float)

data.head()

data.drop(['Mileage','Engine','Power','New_Price'],axis=1,inplace=True)

data.head()

data.shape

data.info()

data.replace({"First":1,"second":2,"Third": 3,"Fourth & Above":4},inplace=True)
data.head()

data.columns

var = 'Fuel_Type'
data[var].value_counts()
Fuel_t =data[[var]]
Fuel_t = pd.get_dummies(Fuel_t,drop_first=True)
Fuel_t.head()

var = 'Transmission'
data[var].value_counts()

Transmission = data[[var]]
Transmission = pd.get_dummies(Transmission,drop_first=True)
Transmission.head()

newdata= pd.concat([data,Fuel_t,Transmission],axis=1)
newdata.head()

newdata.drop(["Fuel_Type","Transmission"],axis=1,inplace=True)
newdata.head()

newdata.corr()



import seaborn as sns
sns.pairplot(newdata)

import matplotlib.pyplot as plt


cormat=newdata.corr()
top_corr=cormat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(newdata[top_corr].corr(),annot=True,cmap='RdYlGn')


x= newdata.loc[:,['Kilometers_Driven','Seats','no_of_years','mileage','engine','power','presentprice','Fuel_Type_Diesel',
                 'Fuel_Type_Petrol','Transmission_Manual']]
x.head()


y=newdata.iloc[:,3]



y.head()

from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(x,y)

print(model.feature_importances_)

from sklearn.model_selection import train_test_split
x_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor()

import numpy as np
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]

from sklearn.model_selection import RandomizedSearchCV
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)

rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
rf.fit(x_train,y_train)
rf.score(x_train,y_train)

rf_random.fit(x_train,y_train)


rf_random.best_params_

rf_random.best_score_

predictions=rf_random.predict(X_test)

sns.distplot(y_test-predictions)

plt.scatter(y_test,predictions)

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

import pickle

file = open('random_forest_regression_modell.pkl', 'wb')
pickle.dump(rf_random, file)



