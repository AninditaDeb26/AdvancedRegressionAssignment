#!/usr/bin/env python
# coding: utf-8

# In[620]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.feature_selection import RFE
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error

import warnings
warnings.filterwarnings('ignore')


# In[521]:


df = pd.read_csv('C:\\Users\\AninditaDeb\\OneDrive\\Documents\\UpgradPGProgram\\AdvancedRegressionAssignment\\train.csv')


# In[522]:


df.info()


# In[523]:


df.head()


# In[524]:


df.shape


# In[525]:


df.isnull().sum()


# In[526]:


df.describe()


# In[527]:


# Separated numeric and categorical features here
df_numeric = df.select_dtypes(include = ['int64', 'float64'])
df_categorical = df.select_dtypes(include = ['object'])


# In[528]:


df_numeric


# In[529]:


df_numeric.columns


# ## Detect Outliers

# In[530]:


percentage_outlier = {}

for feature in df_numeric.columns:
    IQR = df_numeric[feature].quantile(.75) - df_numeric[feature].quantile(.25)
    count_outlier = df_numeric[(df_numeric[feature] > (df_numeric[feature].quantile(.75) + 1.5 * IQR)) | (df_numeric[feature] < (df_numeric[feature].quantile(.25) - 1.5 * IQR))].shape[0]
    percentage_outlier[feature] = round(count_outlier / df_numeric.shape[0] * 100, 2)
    
df_outlier = pd.DataFrame({'Features':list(percentage_outlier.keys()),'Percentage':list(percentage_outlier.values())})
df_outlier.sort_values(by="Percentage", ascending=False)


# #### Numeric values which don't have 0.00 as percentage are having outliers. Dropping all these data will result into data loss, so minimum and maximum values are assigned to the values with outliers.

# In[531]:


for feature, percentage in percentage_outlier.items():
    if feature != 'SalePrice':
        IQR = df[feature].quantile(.75) - df[feature].quantile(.25) 
        max_value = df[feature].quantile(.75) + 1.5 * IQR
        min_value = df[feature].quantile(.25) - 1.5 * IQR
        df[feature][df[feature] > max_value] = max_value
        df[feature][df[feature] < min_value] = min_value


# In[532]:


df.describe()


# In[533]:


df_categorical.info()


# In[534]:


df_numeric.info()


# In[535]:


# Dropping ID column from df_numeric and df
df.drop(['Id'], axis = 1, inplace = True)
df_numeric.drop(['Id'], axis = 1, inplace = True)


# In[536]:


df_numeric.info()


# In[537]:


# Univariate and Bivariate Analysis of numeric features
fig = plt.subplots(figsize = (12, 12))

for i, feature in enumerate(['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1','BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath']):
    plt.subplot(9, 3, i+1)
    sns.scatterplot(x = feature, y = 'SalePrice', data = df)
    plt.tight_layout()


# In[538]:


fig = plt.subplots(figsize = (12, 12))

for i, feature in enumerate(['BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']):
    plt.subplot(9, 3, i+1)
    sns.scatterplot(x = feature, y = 'SalePrice', data = df)
    plt.tight_layout()


# In[539]:


df.drop(['LowQualFinSF', 'KitchenAbvGr', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal'], axis = 1, inplace = True)

df.columns


# In[540]:


fig = plt.subplots(figsize = (12, 12))

for i, feature in enumerate(['GarageYrBlt', 'YearBuilt', 'YearRemodAdd', 'YrSold']):
    plt.subplot(4, 2, i+1)
    sns.scatterplot(x = feature, y = 'SalePrice', data = df)
    plt.tight_layout()


# In[541]:


# Handling missing value

for feature in df.select_dtypes(exclude=['object']).columns:
    if df[feature].isnull().any():
        print(feature, ':', round(df[feature].isnull().sum()/df.shape[0], 2) * 100)


# In[542]:


# Dropping rows with Null values from MasVnrArea as it only has 1% missing values.
df = df[~df['MasVnrArea'].isnull()]


# In[543]:


df.shape


# In[544]:


# Check correlation in data
plt.figure(figsize = (20, 20))
sns.heatmap(df_numeric.corr(), annot = True)
plt.show()


# In[545]:


# Analyzing categorical Features
df_categorical.columns


# In[546]:


# Handling missing value

for feature in df.select_dtypes(include=['object']).columns:
    if df[feature].isnull().any():
        print(feature, ':', round(df[feature].isnull().sum()/df.shape[0], 2) * 100)


# In[547]:


df_categorical_features = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC' ,'Fence' ,'MiscFeature']

df_missing_values = df[df_categorical_features].isnull().sum()

print(df_missing_values)


# In[548]:


#drop PoolQC for number of missing values being high
df.drop(['PoolQC'], axis = 1, inplace = True)

#drop rows with null values for electrical as number of missing value is very less
df.dropna(subset=['Electrical'], inplace = True)


# In[549]:


#Replace missing values with Not_Applicable
df_categorical_features = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'Fence' ,'MiscFeature']

df[df_categorical_features] = df[df_categorical_features].fillna(value = 'Not_Applicable', axis = 1)

df[df_categorical_features].isnull().sum()


# In[550]:


df['BsmtQual'].unique()


# In[551]:


for feature in df.columns:
    if df[feature].isnull().any():
        print(feature, ':', round(df[feature].isnull().sum()/df.shape[0], 2) * 100)


# In[552]:


df.columns.shape


# In[553]:


# Generate boxplot for SalePrice vs features

def generate_boxplot(feature_list):
    fig=plt.subplots(figsize=(20, 12))
    for i, feature in enumerate(feature_list):
        plt.subplot(4, 2, i+1)
        sns.boxplot(x="SalePrice", y=feature, data=df)
        plt.tight_layout()


# In[554]:


# Analyzing ordered features

features1 = ['LotShape', 'Utilities', 'LandSlope', 'HouseStyle', 'ExterQual', 'ExterCond']
generate_boxplot(features1)


# In[555]:


features2 = ['HeatingQC', 'KitchenQual', 'Functional', 'FireplaceQu']
generate_boxplot(features2)


# In[556]:


garage_features = ['GarageFinish', 'GarageQual', 'GarageCond']
generate_boxplot(garage_features)


# In[557]:


basement_features = ['BsmtQual' , 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
generate_boxplot(basement_features)


# In[558]:


# Encoding categorical features
df['LotShape'] = df['LotShape'].map({'Reg':0, 'IR1':1,'IR2':2,'IR3':3})
df['Utilities'] = df['Utilities'].map({'AllPub':0, 'NoSewr':1, 'NoSeWa':2, 'ELO':3})
df['LandSlope'] = df['LandSlope'].map({'Gtl':0,'Mod':1,'Sev':2})
df['HouseStyle'] = df['HouseStyle'].map({'1Story':0, '1.5Fin':1, '1.5Unf':2, '2Story' :3, '2.5Fin':4, '2.5Unf':5, 'SFoyer':6, 'SLvl':7})
df['ExterQual'] = df['ExterQual'].map({'Ex':0, 'Gd':1, 'TA':2, 'Fa':3, 'Po':4})
df['ExterCond'] = df['ExterCond'].map({'Ex':0, 'Gd':1, 'TA':2, 'Fa':3, 'Po':4})
df['BsmtQual'] = df['BsmtQual'].map({'Ex':0, 'Gd':1, 'TA':2, 'Fa':3, 'Po':4, 'Not_Applicable':5})
df['BsmtCond'] = df['BsmtCond'].map({'Ex':0, 'Gd':1, 'TA':2, 'Fa':3, 'Po':4, 'Not_Applicable':5})
df['BsmtExposure'] = df['BsmtExposure'].map({'Gd':0, 'Av':1, 'Mn':2, 'No':3, 'Not_Applicable':4})
df['BsmtFinType1'] = df['BsmtFinType1'].map({'GLQ':0, 'ALQ':1, 'BLQ':2, 'Rec':3, 'LwQ':4, 'Unf':5, 'Not_Applicable':6})
df['BsmtFinType2'] = df['BsmtFinType2'].map({'GLQ':0, 'ALQ':1, 'BLQ':2, 'Rec':3, 'LwQ':4, 'Unf':5, 'Not_Applicable':6})
df['HeatingQC'] = df['HeatingQC'].map({'Ex':0, 'Gd':1, 'TA':2, 'Fa':3, 'Po':4})
df['CentralAir'] = df['CentralAir'].map({'N':0,'Y':1})
df['KitchenQual'] = df['KitchenQual'].map({'Ex':0, 'Gd':1, 'TA':2, 'Fa':3, 'Po':4})
df['GarageFinish'] = df['GarageFinish'].map({'Fin':0, 'RFn':1, 'Unf':2, 'Not_Applicable':3})
df['GarageQual'] = df['GarageQual'].map({'Ex':0, 'Gd':1, 'TA':2, 'Fa':3, 'Po':4, 'Not_Applicable':5})
df['GarageCond'] = df['GarageCond'].map({'Ex':0, 'Gd':1, 'TA':2, 'Fa':3, 'Po':4, 'Not_Applicable':5})
df['Functional'] = df['Functional'].map({'Typ':0, 'Min1':1, 'Min2':2, 'Mod':3, 'Maj1':4, 'Maj2':5, 'Sev':6, 'Sal':7})
df['FireplaceQu'] = df['FireplaceQu'].map({'Ex':0, 'Gd':1, 'TA':2, 'Fa':3, 'Po':4, 'Not_Applicable':5})


# In[559]:


df['BsmtQual'].unique()


# In[560]:


# Checking the features after encoding

df[['LotShape', 'Utilities', 'LandSlope', 'HouseStyle', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual','Functional','FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond']].info()


# In[561]:


#Analyzing unordered features
unordered_features = ['MSZoning', 'Street', 'Alley', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1' , 'Condition2', 
                      'BldgType', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 
                      'Electrical', 'GarageType','PavedDrive', 'Fence','MiscFeature', 'SaleType','SaleCondition']
generate_boxplot(['MSZoning', 'Street', 'Alley', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1' , 'Condition2']) 


# In[562]:


generate_boxplot(['BldgType', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating'])


# In[563]:


generate_boxplot(['Electrical', 'GarageType','PavedDrive', 'Fence','MiscFeature', 'SaleType','SaleCondition'])


# In[564]:


# Encoding Categorical Variqables
df[unordered_features]


# In[565]:


df_dummies = pd.get_dummies(df[unordered_features], drop_first = True)
df_dummies.shape


# In[566]:


# drop features with same values in more than or equal to 95% rows to get rid of imbalanced data
drop_features = []
for feature in df_dummies.columns:
    if df_dummies[feature].value_counts()[0]/df_dummies.shape[0] >= 0.95:
        drop_features.append(feature)
        
print(drop_features)
print(len(drop_features))


# In[567]:


df_dummies = df_dummies.drop(drop_features, axis = 1)


# In[568]:


df_dummies.shape


# In[569]:


df.shape


# In[570]:


# concat df with dummy variables
df = pd.concat([df, df_dummies], axis = 1)
df.shape


# In[571]:


#drop duplicate columns
df = df.drop(unordered_features, axis = 1)
df.shape


# In[572]:


# Splitting train and test dataset

X = df.drop(['SalePrice'], axis = 1)
X.head()


# In[573]:


# plot SalePrice to check the distribution
plt.title('Distribution of SalePrice')
sns.distplot(df['SalePrice'])
plt.show()


# In[574]:


# plot log of SalePrice to check the distribution
plt.title('Distribution of log transformed SalePrice')
sns.distplot(np.log(df['SalePrice']))
plt.show()


# In[575]:


# Transform SalePrice into log to get normally distributed data
y = np.log(df['SalePrice'])
y.head()


# In[576]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state = 100)


# In[577]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[578]:


X['LotFrontage'].isnull().sum()


# In[579]:


feature_LotFrontage = SimpleImputer(missing_values = np.nan, strategy = 'mean')


# In[580]:


feature_LotFrontage.fit(X_train[['LotFrontage']])


# In[582]:


X_train[['LotFrontage']] = feature_LotFrontage.transform(X_train[['LotFrontage']])


# In[583]:


X_test[['LotFrontage']] = feature_LotFrontage.transform(X_test[['LotFrontage']])


# In[584]:


feature_GarageYrBlt = SimpleImputer(missing_values = np.nan, strategy = 'median')


# In[585]:


feature_GarageYrBlt.fit(X_train[['GarageYrBlt']])


# In[586]:


X_train[['GarageYrBlt']] = feature_GarageYrBlt.transform(X_train[['GarageYrBlt']])


# In[587]:


X_test[['GarageYrBlt']] = feature_GarageYrBlt.transform(X_test[['GarageYrBlt']])


# In[588]:


# Feature scaling
X_train.values


# In[589]:


ss = StandardScaler()
ss.fit(X_train)


# In[590]:


X_train_scaled = pd.DataFrame(data = ss.transform(X_train), columns = X_train.columns)
X_test_scaled = pd.DataFrame(data = ss.transform(X_test), columns = X_test.columns)


# In[591]:


print(X_train_scaled)
print(X_test_scaled)


# In[608]:


# Feature Selection using RFE
def top_features(n):
    top_n_cols = []

    lm = LinearRegression()
    lm.fit(X_train_scaled, y_train)
    rfe = RFE(lm, n_features_to_select = n)
    rfe = rfe.fit(X_train_scaled, y_train)

    print("Top %d features : " % n)
    rfe_ranking = list(zip(X_train_scaled.columns, rfe.support_, rfe.ranking_))

    for i in rfe_ranking:
        if i[1]:
            top_n_cols.append(i[0])
    print(top_n_cols)
    return top_n_cols


# In[609]:


missing_columns = X_train_scaled.columns[X_train_scaled.isnull().sum() > 0]
missing_columns


# In[610]:


X_train_scaled['GarageYrBlt'].isnull().sum()


# In[611]:


# Checking top 45, 50 and 55 features
top_45 = top_features(45)
top_50 = top_features(50)
top_55 = top_features(55)


# In[614]:


# This will be used to check adjusted R-square value for top 45 features
X_train_ols = sm.add_constant(X_train[top_45])
linear_regression = sm.OLS(y_train.values.reshape(-1,1), X_train_ols).fit()
print(linear_regression.summary())  


# In[615]:


# This will be used to check adjusted R-square value for top 50 features
X_train_ols = sm.add_constant(X_train[top_50])
linear_regression = sm.OLS(y_train.values.reshape(-1,1), X_train_ols).fit()
print(linear_regression.summary())  


# In[616]:


# This will be used to check adjusted R-square value for top 55 features
X_train_ols = sm.add_constant(X_train[top_55])
linear_regression = sm.OLS(y_train.values.reshape(-1,1), X_train_ols).fit()
print(linear_regression.summary())  


# In[617]:


#top 50 and 55 features have same adjusted r squared value
X_train_rfe = X_train_scaled[top_50]
X_test_rfe = X_test_scaled[top_50]


# In[621]:


# Reusable Code Block for Cross-validation, Model Building and Model Evaluation

def build_model(X_train, y_train, X_test, params, model='ridge'):
    if model == 'ridge':
        estimator_model = Ridge()
    else:
        estimator_model = Lasso()
    model_cv = GridSearchCV(estimator = estimator_model, 
                          param_grid = params, 
                          scoring= 'neg_mean_absolute_error', 
                          cv = 5, 
                          return_train_score=True,
                          verbose = 1)
    model_cv.fit(X_train, y_train)
    alpha = model_cv.best_params_["alpha"]
    print("Optimum alpha for %s is %f" %(model, alpha))
    final_model = model_cv.best_estimator_
    final_model.fit(X_train, y_train)
    y_train_pred = final_model.predict(X_train)
    y_test_pred = final_model.predict(X_test)
    
    # Model Evaluation
    print(model," Regression with ",alpha)
    print("===================================")
    print('R2 score (train) : ',r2_score(y_train,y_train_pred))
    print('R2 score (test) : ',r2_score(y_test,y_test_pred))
    print('RMSE (train) : ', np.sqrt(mean_squared_error(y_train, y_train_pred)))
    print('RMSE (test) : ', np.sqrt(mean_squared_error(y_test, y_test_pred)))
    
    return final_model, y_test_pred


# In[622]:


#Ridge Regression
params = {'alpha': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 
                    9.0, 10.0, 20, 50, 100, 500, 1000 ]}

ridge_final_model, y_test_predicted = build_model(X_train_rfe, y_train, X_test_rfe, params, model='ridge')


# In[623]:


#Lasso Regression
params = {'alpha': [0.000001, 0.00001,0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 500, 1000, 10000]}

lasso_final_model, y_test_predicted = build_model(X_train_rfe, y_train, X_test_rfe, params, model='lasso')


# In[628]:


# Comparing Model Coefficients
model_coefficients = pd.DataFrame(index=X_test_rfe.columns)
model_coefficients.rows = X_test_rfe.columns

model_coefficients['Ridge (alpha=50.0)'] = ridge_final_model.coef_
model_coefficients['Lasso (alpha=0.001)'] = lasso_final_model.coef_
pd.set_option('display.max_rows', None)
model_coefficients


# In[629]:


# Converting the predictions to its original scale (anti log)

test_prediction = np.round(np.exp(y_test_predicted)).astype(int)
print(test_prediction[:5])


# In[631]:


# 50 features ordered by feature importance in Lasso Regression

model_coefficients[['Lasso (alpha=0.001)']].sort_values(by='Lasso (alpha=0.001)', ascending=False)
model_coefficients[['Lasso (alpha=0.001)']].sort_values(by='Lasso (alpha=0.001)', ascending=False).index[:10]


# In[ ]:




