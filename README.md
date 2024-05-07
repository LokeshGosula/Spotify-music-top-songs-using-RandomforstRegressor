# Spotify-music-top-songs-using-RandomforstRegressor
## In this notebook i will try to do all end to end ml  i mean from EDA(Exploratory Data Analysis) to Deploying the Final Trained Model
## Generally the common path of any end to end ML Project work flow is like:

1. Importing libraries
2. Loading the data
3. Data preparation
4. Splitting the features and target(i.e.,independent and dependent features)
5. Model creation
6. Training the model
7. Prediction
8. Deployment
# Aim of the project : To Predicting the Song Popularity by using a dataset from kaggle
## Summary:
####     The file is in csv format which contains 1835 rows and 15 columns.
####     This is a regression problem by help of song_ propularity we can conclude
####     The Dependent varible is song_propularity.
####     Regression is a supervised learning algorthim.
####     We  will train the model by using feature and target and predict the model output with best accuracy
![image.png](attachment:47db147d-e0e9-4497-8993-f39ffeed9967.png)
### step 1:  Importing libraries
##### Install the requried libraries :
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble  import ExtraTreesRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
### Step 2 : Loading the data
### Loading the data we need to import the pandas library by pandas we can read the csv file and convert into Dataframe
import pandas as pd
df = pd.read_csv("C:\\Users\\song_data.csv")
print(df)                                        #  converted into dataframe
#### Showing the index of the column names by using attribute
df.columns
#### Shape attribute displays the number of rows and columns
df.shape
#### info() method describe the nan values and datatypes and there is no null values in this dataframe
#### isna().sum() also other way to print null values
df.info()
df.isna().sum()
#### describe method show the mean and meadian and five number summary values
df.describe()
### Step 3 : Data preparation
#### head() - print the top 5 values of the dataframe
df.head()
df.shape
df.nunique()
#### Filtering the columns
df.song_name.nunique()
##### AS songs name are different it will impact our model if we use song_name columns and can't predict best result and we will drop the song_name  
df.drop("song_name",axis=1,inplace=True)
[[262333,0.005520,0.496,0.682,0.000029,8,0.0589,-4.095,1,0.0294,167.060,4,0.474]]
print(df)
df.columns
#### Splitting Independent and dependent varaiable
X =df.iloc[:,1:]
y = df.iloc[:,0]
print(y)
#### Independent values
print(X)
### Dependent values
print(y)
### Model 1 : Selecting features using sklearn get features
from sklearn.ensemble  import ExtraTreesRegressor
model= ExtraTreesRegressor()
model.fit(X,y)
model.feature_importances_
X.columns
X = X[['song_duration_ms','acousticness','danceability','energy','instrumentalness','liveness','loudness','tempo','audio_valence']]
print(X)
### Model 2: Selecting Features using SelectKBest
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
model = SelectKBest(f_regression,k=6)
model.fit(X,y)
X.columns[model.fit(X,y).get_support()]
#### Step 4: Spliting train and test datasets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
print(X_train)
X_train.shape
 [[193360,0.0325,0.869,0.660,0.000126,0.0651,-6.231,134.525,0.782]]
print(X_test)
print(y_train)
print(y_test)
### Creating a model for predicition the model used is RandomForestRegressor
### Step5: Model Creation
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor()
rf_model.fit(X,y)
### Hyperparmeteric tuning with RandomizedSearchCV
from sklearn.model_selection  import RandomizedSearchCV
import numpy as np
#### number of tress in random forest
n_estimators=[int(x) for x in np.linspace(start=100,stop=1000,num=12)]
print(n_estimators)
#### number of features condiser in each split
max_feature=['auto','sqrt']
#### maximum number of level in tree
max_depth=[int(x) for x in np.linspace(start=5,stop=30,num=6)]
max_depth
### Minimum number samples required to split a node
min_sample_split=[10,20,30]
#### Minimum no of samples required to split a each leaf node
min_sample_leaf=[1,2,5]
hyperparameter_tune = {
    "n_estimators": n_estimators,
    "max_features": max_feature,
    "max_depth": max_depth,
    "min_samples_split": min_sample_split,
    "min_samples_leaf": min_sample_leaf
}
print(hyperparameter_grid)
### scoring is loss function,no of iteration is 5, cross_validation=5,output(i.e., epoches)=2,random_state refers to fixing the data 
model = RandomizedSearchCV(estimator=rf_model, param_distributions=hyperparameter_tune,
                          scoring='neg_mean_squared_error', n_iter=5, cv=5, verbose=2,
                          random_state=8)
import warnings
warnings.filterwarnings("ignore")
### Step 6 : Traning the model
model.fit(X_train,y_train)

model.best_params_
### Step 7 : Model prediction
y_predict=model.predict(X_test)
print(y_predict)
#### Predict the value by giving data of 9 inputs
model.predict( [[203360,0.03525,0.469,0.360,0.000026,0.0951,-7.231,132.525,0.482]])
model.predict( [[193360,0.0325,0.869,0.660,0.000126,0.0651,-6.231,134.525,0.782]])
y_predict[0]
import matplotlib.pyplot as plt
import seaborn as sns
### Data Visualization  :  Uniformally distributed graph and data is trained in better way
plt.figure(figsize=(10,10))
sns.distplot(y_test - y_predict)
plt.show()
### tuning the values to get better more results to view in scatter plot
plt.scatter(y_test,y_predict)
### Metrics values
#### The values of MSE and MAE is high 
from sklearn import metrics
print("MAE:",metrics.mean_absolute_error(y_predict,y_test))
print("MSE:",metrics.mean_squared_error(y_predict,y_test))
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_predict,y_test)))