#!/usr/bin/env python
# coding: utf-8

# # Importing Dependencies

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
warnings.filterwarnings('ignore')


# # Data Collection

# In[3]:


#loading the dataset
import pandas as pd
wine = pd.read_excel('wine_quality.xlsx')


# In[4]:


wine


# In[5]:


#first 5 rows of the dataset
wine.head()


# In[6]:


#last 5 rows of the dataset
wine.tail()


# In[7]:


#checking the number of rows and columns of the dataset
wine.shape


# # Data Cleaning

# In[8]:


# Checking the null values in our dataset
wine.isnull().sum()


# In[9]:


# Dropping the missing values
wine.dropna(inplace=True)
wine


# In[10]:


#reseting the index after dropping missing values
wine.reset_index()


# In[11]:


# Checking the duplicate values from the dataset
wine[wine.duplicated()]


# In[12]:


# Dropping the Duplicate Values
wine.drop_duplicates(inplace=True)
wine


# In[13]:


#reseting the index after dropping duplicate values
wine.reset_index()


# In[14]:


wine.shape


# # Data Analysis and Visuallization

# In[15]:


wine.info()


# In[16]:


wine.describe()


# In[17]:


#percentage of white and red wine
count=wine['type'].value_counts(normalize=True)
count


# In[18]:


fig=plt.figure(figsize=(3,3))
plt.title("Visualization of Types of Wine")
sns.countplot(x="type", data=wine, color='green', alpha=0.5, width=0.5)


# In[19]:


#Checking distribution and outlier for alcohol
plt.figure(4)
plt.subplot(121)
plt.title("Distplot")
sns.distplot(wine['alcohol'])
plt.subplot(122)
plt.title("Boxplot")
wine['alcohol'].plot.box(figsize=(5,3))


# In[20]:


#HISTOGRAM 
fig=plt.figure(figsize=(3,3))
ax1=plt.subplot()
plt.title("Citric Acid Visualization")
plt.xlabel("Values of Citric Acid")
plt.ylabel("Count of citric acid")
ax1.hist(data=wine,x='citric acid', alpha=1)
plt.show()


# In[ ]:


# Wine Quality comparision with different chemical parameters

fig, axes = plt.subplots(3, 3, figsize=(18, 10))
fig.suptitle('Quality comparison with different parameters')
sns.barplot(ax=axes[0, 0], data=wine, x='quality', y='fixed acidity')
sns.barplot(ax=axes[0, 1], data=wine, x='quality', y='volatile acidity')
sns.barplot(ax=axes[0, 2], data=wine, x='quality', y='citric acid')
sns.barplot(ax=axes[1, 0], data=wine, x='quality', y='alcohol')
sns.barplot(ax=axes[1, 1], data=wine, x='quality', y='residual sugar')
sns.barplot(ax=axes[1, 2], data=wine, x='quality', y='pH')
sns.barplot(ax=axes[2, 0], data=wine, x='quality', y='density')
sns.barplot(ax=axes[2, 1], data=wine, x='quality', y='sulphates')
sns.barplot(ax=axes[2, 2], data=wine, x='quality', y='total sulfur dioxide')


# In[23]:


#PIE CHART
num=[74.44, 25.55]
labels=['White Wine', 'Red Wine']
explode=[0,0.05]
fig=plt.figure(figsize=(5,5))
color=['lightgrey','red']
plt.title("Percentage of Production of Wine")
plt.pie(num,labels=labels, explode=explode, shadow=True, colors=color, autopct="%1.1f%%")
plt.legend(title="Types of Wine")
plt.show()


# In[24]:


sns.pairplot(wine)


# # CORRELATION

# In[25]:


#checking correlation
correlation= wine.corr()
correlation


# In[26]:


#buidling heatmap
plt.figure(figsize=(5,5))
sns.heatmap(correlation, 
            cbar=True, 
            square=True, 
            fmt= '.1f', 
            annot=True, 
            annot_kws={'size': 8}, 
            cmap='Greens')


# # Data Preprocessing

# In[27]:


#getting unique values for the type of wine
wine['type'].unique()


# In[28]:


#map function for converting in numerical values
wine['type']=wine['type'].map({'white':0, 'red':1})


# In[29]:


wine


# # Seprating target value from dataset

# In[30]:


X=wine.iloc[::-1]
Y=wine.iloc[:,-1]


# In[31]:


X


# In[32]:


Y


# In[33]:


X = wine.drop('quality',axis=1)


# In[34]:


X


# In[35]:


Y= wine['quality'].apply(lambda y: 1 if y > 7 else 0)
Y


# In[36]:


wine['quality'].value_counts()


# # Standard Scaling

# In[37]:


from sklearn.preprocessing import StandardScaler


# In[38]:


scaler = StandardScaler()
scaler.fit(X)
x_standard = scaler.transform(X)


# In[39]:


x_standard


# # Train & Test Split

# In[40]:


from sklearn.model_selection import train_test_split


# In[41]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=.2,random_state=123)


# In[42]:


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[43]:


X_test


# In[44]:


Y_train


# In[45]:


Y_test


# # MODEL TRAINING

# # Logistic Regression

# In[46]:


from sklearn.linear_model import LogisticRegression


# In[47]:


logreg = LogisticRegression()


# In[48]:


logreg.fit(X_train,Y_train)


# In[49]:


y_pred = logreg.predict(X_test)


# # MODEL EVALUATION

# In[50]:


from sklearn.metrics import accuracy_score


# In[51]:


accuracy_score(Y_test, y_pred)


# In[52]:


from sklearn.metrics import precision_score, recall_score, f1_score


# In[53]:


from sklearn.metrics import classification_report


# In[54]:


print(classification_report(Y_test, y_pred))


# In[55]:


from sklearn.metrics import confusion_matrix


# In[56]:


confusion_matrix(Y_test, y_pred)


# In[57]:


from sklearn.metrics import ConfusionMatrixDisplay


# In[58]:


style.use('classic')
cm = confusion_matrix(Y_test, y_pred, labels=logreg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix= cm, display_labels=logreg.classes_)
disp.plot()


# # MODEL TRAINING

# # Decision Tree

# In[59]:


from sklearn.tree import DecisionTreeClassifier


# In[60]:


dtree = DecisionTreeClassifier()


# In[61]:


dtree.fit(X_train, Y_train)


# In[62]:


dtree_pred = dtree.predict(X_test)


# # MODEL EVALUATION

# In[63]:


dtree_acc = accuracy_score(dtree_pred, Y_test)


# In[64]:


print("Test accuracy: {:.2f}%".format(dtree_acc*100))


# In[65]:


print(classification_report(Y_test, dtree_pred))


# In[66]:


style.use('classic')
cm = confusion_matrix(Y_test, dtree_pred, labels=dtree.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix= cm, display_labels=dtree.classes_)
disp.plot()


# # MODEL TRAINING

# # Random Forest Classifier

# In[67]:


from sklearn.ensemble import RandomForestClassifier


# In[68]:


rfc = RandomForestClassifier()


# In[69]:


rfc.fit(X_train, Y_train)


# # MODEL EVALUATION

# In[70]:


# accuracy on test data
rfc.pred = rfc.predict(X_test)
rfc.accuracy = accuracy_score(rfc.pred, Y_test)


# In[71]:


print('Accuracy : ', rfc.accuracy)


# In[72]:


print(classification_report(Y_test, rfc.pred))


# In[73]:


confusion_matrix(Y_test, rfc.pred)


# In[74]:


style.use('classic')
cm = confusion_matrix(Y_test, rfc.pred, labels=rfc.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix= cm, display_labels=rfc.classes_)
disp.plot()


# In[75]:


rfc.feature_importances_


# In[76]:


pd.Series(rfc.feature_importances_,index=wine.drop('quality',axis=1).columns).plot(kind='barh')


# # Model Deployment

# In[77]:


input_data = (1,6.5,0.2,0.35,7.1,0.81,15.0,107.0,0.9878,3.38,0.9,0.2)

# changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the data as we are predicting the label for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = rfc.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==1):
  print('Good Quality Wine')
else:
  print('Bad Quality Wine')


# In[78]:


input_data = (1, 6.8,0.26,0.42,1.7,0.049,41,122,0.993,3.47,0.48,10.5)


# changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the data as we are predicting the label for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = rfc.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==1):
  print('Good Quality Wine')
else:
  print('Bad Quality Wine')


# In[79]:


x = np.array(["LR", "DT", "RFC"])
y = np.array([95,93,97])
fig=plt.figure(figsize=(5,5))
ax1=plt.subplot()
ax1.barh(x,y, color='g', alpha=1, height = 0.2)
plt.xlabel('Accuracy Score')
plt.ylabel('Model Name')
plt.title('Accuracy Evaluation of different metricies')
#plt.bar(x,y)
plt.show()


# In[ ]:




