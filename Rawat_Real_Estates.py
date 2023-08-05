import pandas as pd
import numpy as np
housing=pd.read_csv("hdata.csv")
housing.head()
housing.info()
housing.describe()

# For plotting histograms
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))

# # Train-Test Splitting

from sklearn.model_selection import train_test_split
train_set, test_set=train_test_split(housing, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}")

from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['X4 number of convenience stores']):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]
 
strat_train_set['X4 number of convenience stores'].value_counts()
strat_test_set['X4 number of convenience stores'].value_counts()

#Before looking for correlations, make sure to create a copy of the data
housing=strat_train_set.copy()

# # Looking for Correlations

corr_matrix=housing.corr()
corr_matrix['Y house price of unit area'].sort_values(ascending=False)

from pandas.plotting import scatter_matrix
attributes=["Y house price of unit area","X4 number of convenience stores","X2 house age","X3 distance to the nearest MRT station"]
scatter_matrix(housing[attributes], figsize=(12,8))

housing.plot(kind='scatter', x='X3 distance to the nearest MRT station', y='Y house price of unit area', alpha=0.8)

# # Trying out Attribute Combinations

housing['Hage_price']=housing['X2 house age']/housing['Y house price of unit area']

housing.head()

corr_matrix=housing.corr()
corr_matrix['Y house price of unit area'].sort_values(ascending=False)

housing.plot(kind='scatter', x='Hage_price', y='Y house price of unit area', alpha=0.8)

housing=strat_train_set.drop("Y house price of unit area", axis=1)
housing_labels=strat_train_set['Y house price of unit area'].copy()


# # Missing attributes 

#To deal with missing values which might be encountered in future

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy='median')
imputer.fit(housing)


imputer.statistics_

X=imputer.transform(housing)
housing_tr=pd.DataFrame(X, columns=housing.columns)
housing_tr.describe()

# ## Feature Scaling

# # Creating a Pipeline

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline=Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    #.....we can add as many as we want in our pipeline
    ('std_scaler', StandardScaler())
    
])

housing_num_tr=my_pipeline.fit_transform(housing_tr)
housing_num_tr

housing_num_tr.shape

# # Selecting a desired model

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#model=LinearRegression()
#model=DecisionTreeRegressor()
model=RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)

some_data=housing.iloc[:5]
some_labels=housing_labels.iloc[:5]

prepared_data=my_pipeline.transform(some_data)
model.predict(prepared_data)

list(some_labels)

# # Evaluating the model

from sklearn.metrics import mean_squared_error
housing_predictions=model.predict(housing_num_tr)
mse=mean_squared_error(housing_labels, housing_predictions)
rmse=np.sqrt(mse)

print(rmse)

# linearRegression : mse- 60
# DecisionTreeRegressor : mse- 0.0 (overfitting: Model has seen the data and didn't learn the trend)

# ## using better evaluation technique- CROSS VALIDATION

# 1 2 3 4 5 6 7 8 9 10
from sklearn.model_selection import cross_val_score
scores=cross_val_score(model, housing_num_tr, housing_labels, scoring='neg_mean_squared_error', cv=10)
rmse_scores=np.sqrt(-scores)

rmse_scores

def print_scores(scores):
    print('scores: ', scores)
    print('mean: ', scores.mean())
    print('standard deviation: ', scores.std())
    
print_scores(rmse_scores)

# # Saving the model

from joblib import dump, load
dump(model, 'Rawat.joblib')

# # Testing the model on Test Data

X_test =strat_test_set.drop("Y house price of unit area", axis=1)
Y_test =strat_test_set['Y house price of unit area'].copy()
X_test_prepared=my_pipeline.transform(X_test)
final_prediction= model.predict(X_test_prepared)
final_mse=mean_squared_error(Y_test, final_prediction)
final_rmse=np.sqrt(final_mse)

final_rmse

print(final_prediction, list(Y_test))
prepared_data[0]

# # USING THE MODEL

import numpy as np
from joblib import dump, load
model= load('Rawat.joblib')

features=np.array([[ 1.33634919, -1.69732575,  0.88998546,  0.88477281, -0.37510886,
       -0.57993566, -0.38876478]])
print(model.predict(features))