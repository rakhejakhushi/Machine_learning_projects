#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install gradio
#!pip install graphviz
#!pip install xgboost


# # User_Driven_House_Price_Prediction

# # Importing the Dependencies

# In[2]:


#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# # Data Collection and Cleaning

# In[3]:


#loading the dataset
data=pd.read_csv("Bengaluru_House_Data.csv")
data.head()


# In[4]:


#last 5 rows of the dataset
data.tail()


# In[5]:


#Number of rows and columns in dataset
data.shape


# In[6]:


data.info()


# In[7]:


#columns in dataset
data.columns


# In[8]:


#unique values of column "area_type"
data['area_type'].unique()


# In[9]:


#counts of unique values
data['area_type'].value_counts()


# # Dropping features that are not required to build the model

# In[10]:


data2 = data.drop(['area_type','society','balcony','availability'],axis='columns')
data2.head()


# In[11]:


data2.describe()


# In[12]:


#checking null values
data2.isnull().sum()


# In[13]:


data2.shape


# In[14]:


#dropping null values
data3 = data2.dropna()
data3.isnull().sum()


# In[15]:


data3.shape


# In[16]:


#checking datatypes of the dataset
data3.dtypes


# In[17]:


#checking duplicates in the dataset
data3[data3.duplicated()]


# In[18]:


#dropping duplicated values
data3.drop_duplicates()
data3


# In[19]:


data3.reset_index()


# # Feature Engineering

# In[20]:


#unique size in the datset
data3['size'].unique()


# In[21]:


#Adding new feature(integer) for bhk (Bedrooms Hall Kitchen)
data3['bhk'] = data3['size'].apply(lambda x: int(x.split(' ')[0]))


# In[22]:


data3.head()


# In[23]:


#unique values in bhk column
data3['bhk'].unique()


# In[24]:


#Explore total_sqft feature
data3['total_sqft'].unique()


# In[25]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[26]:


data3[~data3['total_sqft'].apply(is_float)].head(5)


# Above shows that total_sqft can be a range (e.g. 2100-2850). For such case we can just take average of min and max value 
# in the range. There are other cases such as 34.46Sq. Meter which one can convert to square ft using unit conversion. 
# I am going to just drop such corner cases to keep things simple

# In[27]:


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None   


# In[28]:


convert_sqft_to_num('2100') #returned float value


# In[29]:


convert_sqft_to_num('3090 - 5002')   #returned avg of min and max values


# In[30]:


convert_sqft_to_num('4125Perch')    #returned nothing


# In[31]:


data4 = data3.copy()
data4['total_sqft'] = data4['total_sqft'].apply(convert_sqft_to_num)


# In[32]:


data4.head()


# In[33]:


data4.loc[672]


# In[34]:


#Adding new feature called price per square feet
data5 = data4.copy()
data5['price_per_sqft'] = data5['price']*100000/data5['total_sqft']
data5.head()


# In[35]:


data5_stats = data5['price_per_sqft'].describe()
data5_stats


# In[36]:


#converting into csv file
#data5.to_csv("bhp.csv",index=False)


# In[37]:


#exploring location feature
len(data5['location'].unique())


# # Dimensionality Reduction

# Examine locations which is a categorical variable. 
# We need to apply dimensionality reduction technique here to reduce number of locations

# In[38]:


data5['location'] = data5['location'].apply(lambda x: x.strip())
location_stats = data5.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_stats


# In[39]:


len(location_stats)


# In[40]:


len(location_stats[location_stats<=10])


# In[41]:


len(location_stats[location_stats>10])


# In[42]:


location_stats_lessthan10 = location_stats[location_stats<=10]
location_stats_lessthan10


# In[43]:


len(data5['location'].unique())


# In[44]:


data5['location'] = data5['location'].apply(lambda x: 'Other Location' if x in location_stats_lessthan10 else x)


# In[45]:


len(data5['location'].unique())


# In[46]:


data5.head(10)


# # Outlier Detection & Removal using Business Logic

# As a data scientist when we have a conversation with your business manager (who has expertise in real estate), 
# he will tell you that normally square ft per bedroom is 300 (i.e. 2 bhk apartment is minimum 600 sqft. 
# If you have for example 400 sqft apartment with 2 bhk than that seems suspicious and can be removed as an outlier. 
# We will remove such outliers by keeping our minimum thresold per bhk to be 300 sqft

# In[47]:


data5[data5.total_sqft/data5.bhk<300].head()


# In[48]:


data5.shape


# In[49]:


data6 = data5[~(data5.total_sqft/data5.bhk<300)]


# In[50]:


data6.shape


# # Outlier Removal Using Standard Deviation and Mean

# In[51]:


data6['price_per_sqft'].describe()


# Here we find that min price per sqft is 267 rs/sqft whereas max is 12000000, this shows a wide variation in property prices. 
# We should remove outliers per location using mean and one standard deviation

# In[52]:


def remove_pps_outliers(data):
    data_out = pd.DataFrame()
    for key, subdata in data.groupby('location'):
        m = np.mean(subdata.price_per_sqft)
        st = np.std(subdata.price_per_sqft)
        reduced_data = subdata[(subdata.price_per_sqft>(m-st)) & (subdata.price_per_sqft<=(m+st))]
        data_out = pd.concat([data_out,reduced_data],ignore_index=True)
    return data_out
data7 = remove_pps_outliers(data6)
data7.shape


# #Let's check if for a given location how does the 2 BHK and 3 BHK property prices look like

# In[53]:


def plot_scatter_chart(data,location):
    bhk2 = data[(data.location==location) & (data.bhk==2)]
    bhk3 = data[(data.location==location) & (data.bhk==3)]
    plt.figure(figsize=(15,5))
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(data7,"Rajaji Nagar")


# In[54]:


plot_scatter_chart(data7,"Hebbal")


# In[55]:


plot_scatter_chart(data7,"7th Phase JP Nagar")


# We should also remove properties where for same location, the price of (for example) 3 bedroom apartment is less than 
# 2 bedroom apartment (with same square ft area). What we will do is for a given location, 
# we will build a dictionary of stats per bhk, i.e.
# 
# {
#     '1' : {
#         'mean': 4000,
#         'std: 2000,
#         'count': 34
#     },
#     '2' : {
#         'mean': 4300,
#         'std: 2300,
#         'count': 22
#     },    
# }
# 
# Now we can remove those 2 BHK apartments whose price_per_sqft is less than mean price_per_sqft of 1 BHK apartment

# In[56]:


def remove_bhk_outliers(data):
    exclude_indices = np.array([])
    for location, location_data in data.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_data in location_data.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_data.price_per_sqft),
                'std': np.std(bhk_data.price_per_sqft),
                'count': bhk_data.shape[0]
            }
        for bhk, bhk_data in location_data.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_data[bhk_data.price_per_sqft<(stats['mean'])].index.values)
    return data.drop(exclude_indices,axis='index')
data8 = remove_bhk_outliers(data7)
# data8 = data7.copy()
data8.shape


# # Plot same scatter chart again to visualize price_per_sqft for 2 BHK and 3 BHK properties

# In[57]:


plot_scatter_chart(data8,"Rajaji Nagar")


# In[58]:


plot_scatter_chart(data8,"Hebbal")


# In[59]:


plot_scatter_chart(data8,"7th Phase JP Nagar")


# # Histogram for Price Per Square Feet vs Count

# In[60]:


plt.figure(figsize=(7,5))
plt.hist(data8.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")


# # Outlier Removal Using Bathrooms Feature

# In[61]:


data8.bath.unique()


# In[62]:


data8[data8.bath>10]


# # Histogram for Number of bathrooms vs Count

# In[63]:


plt.figure(figsize=(4,4))
plt.hist(data8.bath,rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")


# It is unusual to have 2 more bathrooms than number of bedrooms in a home

# In[64]:


data8[data8.bath>data8.bhk+2]


# Again the business manager has a conversation with you (i.e. a data scientist) that if you have 4 bedroom home 
# and even if you have bathroom in all 4 rooms plus one guest bathroom, you will have total bath = total bed + 1 max. 
# Anything above that is an outlier or a data error and can be removed

# In[65]:


data9 = data8[data8.bath<data8.bhk+2]
data9.shape


# In[66]:


data9.head()


# In[67]:


data10 = data9.drop(['size','price_per_sqft'],axis='columns')
data10.head()


# # Data Visualization

# In[68]:


data9.hist(figsize=(10,4))


# In[69]:


data.area_type.value_counts().plot(kind='bar')


# In[70]:


sns.distplot(data9["total_sqft"])  
plt.figure(figsize=(4,4))
plt.show()


# In[71]:


sns.pairplot(data9)  
plt.show()


# In[72]:


correlation_matrix = data9.corr()
sns.heatmap(correlation_matrix, annot=True)  # Annotate with correlation values
plt.show()


# In[73]:


plt.figure(figsize=(4, 4))
plt.scatter(data9["price"], data9["bhk"])
plt.xlabel("Price")
plt.ylabel("BHK")
plt.title("Scatter Plot of Price & BHK")
plt.show()


# # Use of One Hot Encoding for Location

# In[74]:


dummies = pd.get_dummies(data10.location)
dummies


# In[75]:


data11 = pd.concat([data10, dummies.drop('Other Location', axis='columns')],axis = 'columns')
data11.head(5)


# In[76]:


data12 = data11.drop('location',axis='columns')
data12.head()


# In[77]:


data12.shape


# # Model Building

# In[78]:


X = data12.drop(['price'],axis=1)
X.head(3)


# In[79]:


X.shape


# In[80]:


y = data12.price
y.head(3)


# In[81]:


len(y)


# # Linear regression implementation

# In[82]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[83]:


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[84]:


# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)


# In[85]:


model.score(X_test,y_test)


# In[86]:


# Make predictions on test data
y_pred = model.predict(X_test)
y_pred


# In[87]:


# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # Root Mean Squared Error
print(f"Root Mean Squared Error (RMSE): {rmse}")


# In[88]:


from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)


# # K Fold cross validation to measure accuracy of our LinearRegression model

# In[89]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score


# In[90]:


cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
cross_val_score(LinearRegression(), X, y, cv=cv)


# We can see that in 5 iterations we get a score above 80% all the time. 
# This is pretty good but we want to test few other algorithms for regression to see if we can get even better score. 
# We will use GridSearchCV for this purpose

# # Lasso Regression Implementation

# In[91]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[92]:


# Replace with the appropriate hyperparameter (alpha) for regularization strength
lasso_regressor = Lasso(alpha=0.1)  # Adjust alpha as needed


# In[93]:


lasso_regressor.fit(X_train, y_train)


# In[94]:


y_pred_lasso = lasso_regressor.predict(X_test)
y_pred_lasso


# In[95]:


y_pred_lasso = lasso_regressor.predict(X_test)


# In[96]:


y_pred_lasso


# In[97]:


# Calculate MSE
mse = mean_squared_error(y_test, y_pred_lasso)

# Calculate RMSE (square root of MSE)
rmse = np.sqrt(mse)

# Calculate MAE
mae = mean_absolute_error(y_test, y_pred_lasso)

# Print the results
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)


# # Decision Tree Implementation

# In[98]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[99]:


# Feature scaling (optional but recommended)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  


# In[100]:


# Create a Decision Tree Regressor
decision_tree = DecisionTreeRegressor(max_depth=5)  # Adjust max_depth as needed


# In[101]:


#Train the model
decision_tree.fit(X_train_scaled, y_train)


# In[102]:


# Make predictions on the test set
y_pred = decision_tree.predict(X_test_scaled)
y_pred


# In[103]:


# Evaluate model performance
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} Lakhs")


# # XGBoost Implementation

# In[104]:


from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


# In[105]:


# Create XGBoost model
model = XGBRegressor()
model


# In[106]:


# Train the model
model.fit(X_train, y_train)


# In[107]:


# Make predictions on test set
y_pred = model.predict(X_test)
y_pred


# In[108]:


# Evaluate model performance
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} Lakhs")


# In[109]:


from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): ₹{mae:.2f} Lakhs")


# In[110]:


from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2:.2f}")


# # SVM Implementation

# In[111]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error


# In[112]:


# Feature scaling (SVM is sensitive to scale)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[113]:


# Create SVR model
model = SVR(kernel='rbf') 
model


# In[114]:


# Train the model
model.fit(X_train_scaled, y_train)


# In[115]:


# Make predictions on test set
y_pred = model.predict(X_test_scaled)
y_pred


# In[116]:


# Evaluate model performance
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} Lakhs")


# # Random Forest Classification

# In[117]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# In[118]:


# Create Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100)
model


# In[119]:


# Train the model
model.fit(X_train, y_train)


# In[120]:


# Make predictions on test set
y_pred = model.predict(X_test)
y_pred


# In[121]:


# Evaluate model performance
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} Lakhs")


#   # Find best model using GridSearchCV

# In[122]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor


# In[123]:


import pandas as pd
from sklearn.model_selection import GridSearchCV, ShuffleSplit

def find_best_model_using_gridsearchcv(X, y):
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                'positive': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['mse', 'friedman_mse'],
                'splitter': ['best', 'random']
            }
        }
    }

    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])



# In[124]:


find_best_model_using_gridsearchcv(X,y)


# Based on above results we can say that LinearRegression gives the best score. Hence we will use that.

# # Test the model for few properties

# In[125]:


X.columns


# In[126]:


def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]


# In[127]:


predict_price('1st Phase JP Nagar',1000, 2, 2)


# In[128]:


predict_price('1st Phase JP Nagar',1000, 3, 3)


# In[129]:


predict_price('Indira Nagar',1000, 3, 3)


# # Export the tested model to a pickle file

# In[130]:


import pickle
with open('banglore_home_prices_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)


# # Export location and column information to a file that will be useful later on in our prediction application

# In[131]:


import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))


# In[132]:


import joblib


# In[133]:


joblib.dump(predict_price, "banglore_home_prices_model")


# In[134]:


model = joblib.load("banglore_home_prices_model")


# In[135]:


model


# # Model Deployment

# In[136]:


import gradio as gr
import pandas as pd
import numpy as np
import pickle
import json


# In[137]:


with open('banglore_home_prices_model.pickle', 'rb') as f:
    model = pickle.load(f)

# Load the column information
with open("columns.json", "r") as f:
    columns = json.load(f)


# In[138]:


def predict_price(location, sqft, bath, bhk):
    loc_index = columns['data_columns'].index(location.lower())
    x = np.zeros(len(columns['data_columns']))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return model.predict([x])[0]


# In[139]:


iface = gr.Interface(fn=predict_price, 
                     inputs=["text", "number", "number", "number"], 
                     outputs="text",
                     title="User Driven House Price Prediction Model",
                     examples=[["1st Phase JP Nagar", 1000, 2, 2]])



# Launch the interface
iface.launch()


# In[ ]:




