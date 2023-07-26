#!/usr/bin/env python
# coding: utf-8

# # Rawat Real Estate- Price Predictor

# In[1]:


import pandas as pd


# In[2]:


housing=pd.read_csv("hdata.csv")


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing.describe()


# In[6]:


#get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


# For plotting histograms
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))


# # Train-Test Splitting

# In[8]:


# For learning purpose(the below code can be used by using scikitlearn library)
import numpy as np
def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled=np.random.permutation(len(data))
    print(shuffled)
    test_set_size=int(len(data)*test_ratio)
    test_indices=shuffled[:test_set_size]
    train_indices=shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[9]:


# train_set, test_set = split_train_test(housing, 0.2)


# In[10]:


# print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}")


# In[11]:


from sklearn.model_selection import train_test_split
train_set, test_set=train_test_split(housing, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}")


# In[12]:


#Suppose we have a label say CHAS. It consists of values either 0 or 1. It contains 93 '0' and 7 '1'.
# Now, there are chances that while splitting the data, training data got only '0' as value of CHAS.
# Then testing data will never show '1' as it doesn't about it.
# To make our model familiar with all the possible values we use the following function.....



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

# In[13]:


corr_matrix=housing.corr()
corr_matrix['Y house price of unit area'].sort_values(ascending=False)


# In[14]:


from pandas.plotting import scatter_matrix
attributes=["Y house price of unit area","X4 number of convenience stores","X2 house age","X3 distance to the nearest MRT station"]
scatter_matrix(housing[attributes], figsize=(12,8))


# In[15]:


housing.plot(kind='scatter', x='X3 distance to the nearest MRT station', y='Y house price of unit area', alpha=0.8)


# # Trying out Attribute Combinations

# In[16]:


housing['Hage_price']=housing['X2 house age']/housing['Y house price of unit area']


# In[17]:


housing.head()


# In[18]:


corr_matrix=housing.corr()
corr_matrix['Y house price of unit area'].sort_values(ascending=False)


# In[19]:


housing.plot(kind='scatter', x='Hage_price', y='Y house price of unit area', alpha=0.8)


# In[20]:


housing=strat_train_set.drop("Y house price of unit area", axis=1)
housing_labels=strat_train_set['Y house price of unit area'].copy()


# # Missing attributes

# In[21]:


# In case, we have a dataset in which some values are missing, we can do one of the following:
#     1. Get rid of the missing data points.
#     2. Get rid of the whole attribute.
#     3. Set the value to some value(0, mean or median).


# In[22]:


#a=housing.dropna(subset=['RM'])            #OPTION 1
#housing.drop("RM", axis=1)              #OPTION 2

#median= housing['RM'].median()           #OPTION 3
#housing['RM'].fillna(median)


# In[23]:


#To deal with missing values which might be encountered in future


# In[24]:


from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy='median')
imputer.fit(housing)


# In[25]:


imputer.statistics_


# In[26]:


X=imputer.transform(housing)
housing_tr=pd.DataFrame(X, columns=housing.columns)
housing_tr.describe()


# # Scikit-learn DESIGN

# Primarily, three types of objects:
# 1. Estimators - It estimates some parameter based on a dataset. E.g. imputer. It has a fit method and transform method. fit method: fits the dataset and calculate internal parameters.
# 
# 2. Transformers - transform method takes input and return output based on the learnings from fit(). It also has a convenience function called fit_transform which fits and then transforms.
# 
# 3. Predictors - LinearRegression model (EXAMPLE). fit() and predict() are two common functions. It also gives score() function which evaluates the predictions.

# ## Feature Scaling

# primarily, two types of feature scaling methods:
# 
# 1. Min-Max Scaling (NORMALIZATION) 
#    (value-min)/(max-min)                #will get values b/w 0 and 1 
#    Sklearn provides a class called MinMaxScaler for this.
# 
# 2. Standardization 
#     (value-mean)/(std. deviation)               #makes variance 1 
#     Sklearn provides a class called StandardScaler for this.
#     

# # Creating a Pipeline

# In[27]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline=Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    #.....we can add as many as we want in our pipeline
    ('std_scaler', StandardScaler())
    
])


# In[28]:


housing_num_tr=my_pipeline.fit_transform(housing_tr)
housing_num_tr


# In[29]:


housing_num_tr.shape


# # Selecting a desired model

# In[30]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#model=LinearRegression()
#model=DecisionTreeRegressor()
model=RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)


# In[31]:


some_data=housing.iloc[:5]
some_labels=housing_labels.iloc[:5]


# In[32]:


prepared_data=my_pipeline.transform(some_data)
model.predict(prepared_data)


# In[33]:


list(some_labels)


# # Evaluating the model

# In[34]:


from sklearn.metrics import mean_squared_error
housing_predictions=model.predict(housing_num_tr)
mse=mean_squared_error(housing_labels, housing_predictions)
rmse=np.sqrt(mse)


# In[35]:


rmse


# linearRegression : mse- 60
# DecisionTreeRegressor : mse- 0.0 (overfitting: Model has seen the data and didn't learn the trend)

# ## using better evaluation technique- CROSS VALIDATION

# In[36]:


# 1 2 3 4 5 6 7 8 9 10
from sklearn.model_selection import cross_val_score
scores=cross_val_score(model, housing_num_tr, housing_labels, scoring='neg_mean_squared_error', cv=10)
rmse_scores=np.sqrt(-scores)


# In[37]:


rmse_scores


# In[38]:


def print_scores(scores):
    print('scores: ', scores)
    print('mean: ', scores.mean())
    print('standard deviation: ', scores.std())
    


# In[39]:


print_scores(rmse_scores)


# # Saving the model

# In[40]:


from joblib import dump, load
dump(model, 'Rawat.joblib')


# # Testing the model on Test Data

# In[41]:


X_test =strat_test_set.drop("Y house price of unit area", axis=1)
Y_test =strat_test_set['Y house price of unit area'].copy()
X_test_prepared=my_pipeline.transform(X_test)
final_prediction= model.predict(X_test_prepared)
final_mse=mean_squared_error(Y_test, final_prediction)
final_rmse=np.sqrt(final_mse)


# In[42]:


final_rmse


# In[43]:


print(final_prediction, list(Y_test))


# In[ ]:





# In[44]:


prepared_data[0]


# # USING THE MODEL

# In[45]:


import numpy as np
from joblib import dump, load
model= load('Rawat.joblib')



# In[46]:


features=np.array([[ 1.33634919, -1.69732575,  0.88998546,  0.88477281, -0.37510886,
       -0.57993566, -0.38876478]])
print(model.predict(features))


# In[ ]:




