import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import Utils as util
from scipy.stats import pearsonr

dataframe = pd.read_excel('Participants_Data_Used_Cars/Data_Train.xlsx')
testframe = pd.read_excel('Participants_Data_Used_Cars/Data_Test.xlsx')

# Cleaning data

dataset = util.clean_data(dataframe)
testset = util.clean_data(testframe)

X = dataset.loc[:,['Brand','Location','Transmission','Owner_Type','Year','Kilometers_Driven','Fuel_Type','Mileage','Engine','Power','Seats']]
Y = dataset.loc[:,'Price']

X_test = testset.loc[:,['Brand','Location','Transmission','Owner_Type','Year','Kilometers_Driven','Fuel_Type','Mileage','Engine','Power','Seats']]

# Label encode categories

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
# Encoding only categorical variables
for col in ['Fuel_Type','Transmission','Owner_Type','Location','Brand']:
    # Using whole data to form an exhaustive list of levels
    data=X[col].append(X_test[col])
    le.fit(data.values)
    X[col]=le.transform(X[col])
    X_test[col]=le.transform(X_test[col])

# Handle missing values

""" from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imp = IterativeImputer(max_iter=10, random_state=0)
X = imp.fit_transform(X)
 """
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
X = imp.fit_transform(X)
X_test = imp.fit_transform(X_test)

correlations = {}
for f in [4,5,7,8,9,10]:
    x1 = X[:,f]
    x2 = dataframe['Price'].values
    key = str(f) + ' vs ' + 'price'
    correlations[key] = pearsonr(x1,x2)[0]

data_correlations = pd.DataFrame(correlations, index=['Value']).T
data_correlations.loc[data_correlations['Value'].abs().sort_values(ascending=False).index]

# One Hot encoding categories
from sklearn.preprocessing import OneHotEncoder
data=np.vstack([X, X_test])
ohe = OneHotEncoder(categorical_features=[0,1,2,3])

ohe.fit(data)
X=ohe.transform(X)
X_test=ohe.transform(X_test)



# Splitting the dataset into the Training set and validation set
from sklearn.model_selection import train_test_split
X_train, X_validate, Y_train, Y_validate = train_test_split(X, Y, test_size = 0.1, random_state = 0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
print('Intercept: \n', regressor.intercept_)
print('Coefficients: \n', regressor.coef_)
# Predicting for validation set
Y_pred = regressor.predict(X_validate)

# with statsmodels
X_train = sm.add_constant(X) # adding a constant
model = sm.OLS(Y, X).fit()
predictions = model.predict(X_validate) 
 
print_model = model.summary()
print(print_model)

# Compare validation set prediction vs actual
df = pd.DataFrame({'Actual': Y_validate, 'Predicted': Y_pred})  

from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_validate, Y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(Y_validate, Y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_validate, Y_pred))) 
print('Root Mean Squared log Error:', np.sqrt(metrics.mean_squared_log_error(Y_validate, Y_pred))) 

Y_test = regressor.predict(X_test)
result_df = pd.DataFrame({'Price':Y_test})

## save to xlsx file
filepath = 'results.xlsx'
result_df.to_excel(filepath, index=False)

