import pandas as pd
import numpy as np
import importlib
import xgboost
import Utils as util
#importlib.reload(util)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import pearsonr
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer

df = pd.read_excel('Participants_Data_Used_Cars/Data_Train.xlsx')
tf = pd.read_excel('Participants_Data_Used_Cars/Data_Test.xlsx')


df = util.clean_data(df)
tf = util.clean_data(tf)

all_features = ['Brand', 'Model', 'Location', 'Owner_Type', 'Transmission', 'Fuel_Type', 'Engine', 'Power', 'Seats', 'Year', 'Kilometers_Driven']

numeric_features = ['Engine', 'Power', 'Seats', 'Year', 'Kilometers_Driven']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))
#    ('scaler', StandardScaler())
    ])

categorical_features = ['Brand', 'Location', 'Owner_Type', 'Transmission', 'Fuel_Type']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
 #       ('vect', CountVectorizer(), 'Model')
        # ('scale', StandardScaler(), all_features)
#        ('iter', IterativeImputer(max_iter=10, random_state=0), ['New_Price'])
        ])



""" clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LinearRegression())])
 """
X = df.drop(['Price', 'Location_Type', 'Name', 'Mileage','New_Price'], axis=1)
y = df['Price']

X_test = tf.drop(['Location_Type', 'Name'], axis=1)

all_data = X.append(X_test)
preprocessor.fit(all_data)

X = preprocessor.transform(X)
X_test = preprocessor.transform(X_test)
X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.1)

xgb = xgboost.XGBRegressor(n_estimators=200, learning_rate=0.08, gamma=0, subsample=0.55,
                           colsample_bytree=0.75, max_depth=20)

xgb.fit(X_train,y_train)
Y_pred = xgb.predict(X_validate)

xgb.fit(X,y)
Y_test = xgb.predict(X_test)

#Initializing Linear regressor
""" from sklearn.linear_model import LinearRegression
lr = LinearRegression()

#Fitting the regressor with training data
lr.fit(X_train,y_train)

#Predicting the target(Price) for predictors in validation set X_val
Y_pred = lr.predict(X_validate)
 """
#Eliminating negative values in prediction for score calculation
for i in range(len(Y_pred)):
   if Y_pred[i] < 0:
       Y_pred[i] = 0

#Printing the score for validation sets
y_true = y_validate
print("\n\n Linear Regression SCORE : ", util.score(Y_pred, y_true))
# clf.fit(X_train, y_train)

# print("model score: %.3f" % clf.score(X_validate, y_validate))

""" 
#Initializing Linear regressor 2
lr2 = LinearRegression()

#Fitting the regressor with training data
lr2.fit(X,y)

#Predicting the target(Price) for predictors in validation set X_val
Y_test = lr2.predict(X_test)

#Eliminating negative values in prediction for score calculation
for i in range(len(Y_test)):
   if Y_test[i] < 0:
       Y_test[i] = 0

result_df = pd.DataFrame({'Price':Y_test})

## save to xlsx file
filepath = 'results_3.xlsx'
result_df.to_excel(filepath, index=False)
 """

