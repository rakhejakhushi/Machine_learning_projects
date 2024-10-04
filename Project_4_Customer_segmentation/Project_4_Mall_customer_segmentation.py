#!/usr/bin/env python
# coding: utf-8

# # Importing the dependencies

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


# # Data collection and Cleaning

# In[2]:


#loading the dataset
customer= pd.read_excel("Mall_Customers.xlsx")
customer


# In[3]:


customer


# In[4]:


#first five rows of the dataset
customer.head()


# In[5]:


customer.shape


# In[6]:


#information regarding dataset
customer.info()


# In[7]:


#checking for null values
customer.isnull().sum()


# In[8]:


#checking for duplicate values
customer[customer.duplicated()]


# In[9]:


customer.describe()


# In[10]:


customer.dtypes


# In[11]:


#dropping "CustomerID" column
customer.drop(["CustomerID"],axis=1, inplace=True)
customer


# # Data Analysis and Visualization

# In[12]:


#Count Plot for Annual Income
sns.set_style('whitegrid')
plt.figure(figsize=(15, 5))
sns.countplot(x="Annual Income (k$)", data=customer, palette='husl');


# In[13]:


# Plot distribution for each column in dataset
plt.figure(1, figsize=(10,5))
n=0
for x in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
    n += 1
    plt.subplot(1, 3, n)
    plt.subplots_adjust(hspace=0.5, wspace= 0.5)
    sns.distplot(customer[x], bins= 20)
    plt.title("Distplot of {}".format(x))
plt.show()


# In[15]:


# Distribution of male and females in the dataset
values = customer['Gender'].value_counts()
labels = ['Male', 'Female']
fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
explode = (0, 0.06)
plt.title("Distribution of Male and female")
patches, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.2f%%', shadow=True,startangle=90, explode=explode)
plt.setp(texts, color='black')
plt.setp(autotexts, size=12, color='white')
autotexts[1].set_color('black')
plt.legend(title="Gender")
plt.show()


# In[71]:


# Age Distribution
plt.figure(figsize=(3,3))
sns.histplot(customer['Age'], bins=10, kde=True, color='pink')
plt.title('Histogram for Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[72]:


# Distribution of Annual income
plt.figure(figsize=(3,3))
sns.kdeplot(customer['Annual Income (k$)'], shade=True, color='skyblue')
plt.title('Distribution of Annual Income')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Density')
plt.show()


# In[19]:


# Box plot for Spending Score across Genre
plt.figure(figsize=(4,3))
sns.boxplot(x='Gender', y='Spending Score (1-100)', data=customer, palette='pastel')
plt.title('Box plot for Spending Score across Gender')
plt.ylabel('Spending Score (1-100)')
plt.xlabel('Gender')
plt.show()


# In[20]:


# Box Plot for Annual Income across Gender
plt.figure(figsize=(4,3))
sns.boxplot(x='Gender', y='Annual Income (k$)', data=customer, palette='pastel')
plt.title('Box plot for Annual Income (k$) across Gender')
plt.ylabel('Annual Income (k$)')
plt.xlabel('Gender')
plt.show()


# In[23]:


# Comparision of Spending Score of Male and Female
plt.figure(figsize=(10,4))
sns.histplot(data=customer, x='Spending Score (1-100)', hue='Gender', binwidth=5, multiple='stack');


# In[24]:


# Distribution of Males in the dataset
fig, ax = plt.subplots(1, 3, figsize=(20, 5))
data = customer.groupby(by='Gender')
data.get_group("Male").plot(kind='hist', ax=ax, subplots=True, bins=40);


# In[69]:


#sns.pairplot(data.get_group("Male"));


# In[26]:


# Distribution of Females in the dataset
fig, ax = plt.subplots(1, 3, figsize=(20, 5))
data = customer.groupby(by='Gender')
data.get_group("Female").plot(kind='hist', ax=ax, subplots=True, bins=40);


# In[70]:


#sns.pairplot(data.get_group("Female"));


# In[28]:


# Voilin Distribution for Age, Annual Income & spending Score based on Gender
plt.figure(1, figsize=(10,4))
n=0
for cols in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
    n += 1
    plt.subplot(1, 3, n)
    sns.set(style = "whitegrid")
    plt.subplots_adjust(hspace=0.5, wspace= 0.5)
    sns.violinplot(x = cols, y = 'Gender', data = customer)
    plt.ylabel('Gender' if n == 1 else " ")
    plt.title("Violin Plot")
plt.show()


# In[73]:


# Analysing Spending Score on the basis of Age 
age_18_25 = customer.Age[(customer.Age >= 18) & (customer.Age <= 25)]
age_26_35 = customer.Age[(customer.Age >= 26) & (customer.Age <= 35)]
age_36_45 = customer.Age[(customer.Age >= 36) & (customer.Age <= 45)]
age_46_55 = customer.Age[(customer.Age >= 46) & (customer.Age <= 55)]
age_55above = customer.Age[customer.Age >= 56]
agex = ["18-25", "26-35", "36-45", "46-55", "55+"]
agey = [len(age_18_25.values), len(age_26_35.values), len(age_36_45.values), len(age_46_55.values), len(age_55above.values)]
plt.figure(figsize=(5,4))
sns.barplot(x= agex, y= agey, palette = "rocket")
plt.title("Analysisng Spending Score on the Basis of Age")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.show()


# In[31]:


# Scatter Plot for Annual income & Spending score
plt.figure(figsize=(5, 4))
sns.scatterplot(x="Annual Income (k$)", y="Spending Score (1-100)", hue="Gender", data=customer)
plt.title("Scatter Plot for Annual income & Spending score")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()


# In[74]:


# Analysing Spending score on the basisi of Customer Scores
ss_1_20 = customer["Spending Score (1-100)"][(customer["Spending Score (1-100)"] >= 1) & (customer["Spending Score (1-100)"] <= 20)]
ss_21_40 = customer["Spending Score (1-100)"][(customer["Spending Score (1-100)"] >= 21) & (customer["Spending Score (1-100)"] <= 40)]
ss_41_60 = customer["Spending Score (1-100)"][(customer["Spending Score (1-100)"] >= 41) & (customer["Spending Score (1-100)"] <= 60)]
ss_61_80 = customer["Spending Score (1-100)"][(customer["Spending Score (1-100)"] >= 61) & (customer["Spending Score (1-100)"] <= 80)]
ss_81_100 = customer["Spending Score (1-100)"][(customer["Spending Score (1-100)"] >= 81) & (customer["Spending Score (1-100)"] <= 100)]
ssx = ["1-20", "21-40", "41-60", "61-80", "81-100"]
ssy = [len(ss_1_20.values), len(ss_21_40.values), len(ss_41_60.values), len(ss_61_80.values), len(ss_81_100.values)]
plt.figure(figsize=(5,4))
sns.barplot(x= ssx, y= ssy, palette = "mako")
plt.title("Analysisng Spending Score on the Basis of Customer Score")
plt.xlabel("Score")
plt.ylabel("Number of Customers Having the Score")
plt.show()


# In[75]:


# Analysing Annual income on the Basis of Customer's Income
ai_0_30 = customer['Annual Income (k$)'][(customer['Annual Income (k$)'] >= 0) & (customer['Annual Income (k$)'] <= 30)]
ai_31_60 = customer['Annual Income (k$)'][(customer['Annual Income (k$)'] >= 31) & (customer['Annual Income (k$)'] <= 60)]
ai_61_90 = customer['Annual Income (k$)'][(customer['Annual Income (k$)'] >= 61) & (customer['Annual Income (k$)'] <= 90)]
ai_91_120 = customer['Annual Income (k$)'][(customer['Annual Income (k$)'] >= 91) & (customer['Annual Income (k$)'] <= 120)]
ai_121_150 = customer['Annual Income (k$)'][(customer['Annual Income (k$)'] >= 121) & (customer['Annual Income (k$)'] <= 150)]
aix = ["0-30", "31-60", "61-90", "91-120", "121-150"]
aiy = [len(ai_0_30.values), len(ai_31_60.values), len(ai_61_90.values), len(ai_91_120.values), len(ai_121_150.values)]
plt.figure(figsize=(5,4))
sns.barplot(x= aix, y= aiy, palette = "Spectral")
plt.title("Analysing Annual Income on the Basis of Customomer\'s income")
plt.xlabel("Income in $ Thousands")
plt.ylabel("Number of Customers")
plt.show()


# # Model Building- K-Means Clustering

# In[64]:


#choosing the annual income and spending score column
X = customer.iloc[:,[2,3]].values
X


# In[65]:


X


# # Finding the number of clusters

# In[66]:


#finding the wcss value for different number of clusters
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


# In[68]:


#Plot the Elbow graph
sns.set()
plt.plot(range(1,11),wcss)
plt.figure(figsize=(2,2))
plt.title("The elbow graph")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()


# # Optimum Number of Clusters will be 5

# In[42]:


# Training the K-means Clustering Model
kmeans = KMeans(n_clusters = 5, random_state = 0)

#returns the label for each data point based on their clusters
Y= kmeans.fit_predict(X)


# In[43]:


Y


# In[44]:


customer['KMeans_Cluster'] = Y


# # Visualizing all the Clusters

# In[45]:


#plotting all the clusters
plt.figure(figsize=(10, 4))
plt.scatter(X[Y==0,0], X[Y==0,1], s=30, c='green', label='Cluster 1')
plt.scatter(X[Y==1,0], X[Y==1,1], s=30, c='red', label='Cluster 2')
plt.scatter(X[Y==2,0], X[Y==2,1], s=30, c='blue', label='Cluster 3')
plt.scatter(X[Y==3,0], X[Y==3,1], s=30, c='yellow', label='Cluster 4')
plt.scatter(X[Y==4,0], X[Y==4,1], s=30, c='violet', label='Cluster 5')

#plotting the centroids
sns.scatterplot(x=kmeans.cluster_centers_[:, 0], y=kmeans.cluster_centers_[:, 1], color='black', s=100, marker='X', label='Centroids')
plt.title('Customer Clusters')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# In[46]:


kmeans.predict([[15,39]])


# # Hierarchical Clustering (Agglomerative Clustering)

# In[47]:


from sklearn.cluster import AgglomerativeClustering


# In[48]:


agg_clustering = AgglomerativeClustering(n_clusters=5)
agg_labels = agg_clustering.fit_predict(X)


# In[49]:


customer['Agg_Cluster'] = agg_labels


# In[50]:


plt.figure(figsize=(10,4))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Agg_Cluster', data=customer, palette='Set1')
plt.title('Hierarchical Clustering (Agglomerative Clustering)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# In[51]:


from scipy.cluster.hierarchy import dendrogram, linkage


# In[52]:


Z = linkage(X, method='ward')


# In[53]:


plt.figure(figsize=(10, 8))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data point')
plt.ylabel('Distance')
plt.show()


# # Saving the Model with Joblib

# In[54]:


import joblib


# In[55]:


joblib.dump(kmeans, "Mall Customer Segmentation")


# In[56]:


model = joblib.load("Mall Customer Segmentation")


# In[57]:


model


# In[58]:


model.predict([[15,39]])


# # Graphical User Interface

# In[59]:


from tkinter import*
import joblib


# In[60]:


#Gui 1
def show_entry_fields():
    a1=int(b1.get())
    a2=int(b2.get())
    
    model = joblib.load("Mall Customer Segmentation")
    result=model.predict([[a1,a2]])
    print("This Customer belongs to Cluster: ", result[0])
    
    if result[0] == 0:
        Label(master, text = "Customers with medium annual income and medium spending score")
    elif result[0] == 1:
        Label(master, text = "Customers with high annual income but low spending score")
    elif result[0] == 2:
        Label(master, text = "Customers with low annual income but low spending score")
    elif result[0] == 3:
        Label(master, text = "Customers with low annual income but high spending score")
    elif result[0] == 4:
        Label(master, text = "Customers with high annual income and high spending score")
        
master = Tk()
master.title("Customer Segmentation")

label=Label(master,text="Customer Segmentation"
            , bg="Black", fg="White").grid(row=0,columnspan=2)

label=Label(master,text="Annual income").grid(row=1)
label=Label(master,text="Spending Score").grid(row=2)

b1 = Entry(master)
b2 = Entry(master)

b1.grid(row=1, column=1)
b2.grid(row=2, column=1)

Button(master, text= "Predict", command = show_entry_fields).grid()

mainloop()


# In[61]:


#GUI 2
from tkinter import Tk, Label, Entry, Button, mainloop
import joblib

def show_entry_fields():
    a1 = int(b1.get())
    a2 = int(b2.get())

    model = joblib.load("Mall Customer Segmentation")
    result = model.predict([[a1, a2]])
    print("This Customer belongs to Cluster:", result[0])

    # Update a Text widget with the prediction result
    result_text.delete(1.0, "end")
    result_text.insert("end", get_cluster_description(result[0]))

def get_cluster_description(cluster):
    descriptions = [
        "Customers with medium annual income and medium spending score",
        "Customers with high annual income but low spending score",
        "Customers with low annual income but low spending score",
        "Customers with low annual income but high spending score",
        "Customers with high annual income and high spending score"
    ]
    return descriptions[cluster]

master = Tk()
master.title("Customer Segmentation")

Label(master, text="Customer Segmentation", bg="black", fg="white").grid(row=0, columnspan=2)
Label(master, text="Annual income").grid(row=1)
Label(master, text="Spending Score").grid(row=2)

b1 = Entry(master)
b2 = Entry(master)

b1.grid(row=1, column=1)
b2.grid(row=2, column=1)

Button(master, text="Predict", command=show_entry_fields).grid(row=3, columnspan=2)

# Add a Text widget to display the prediction result
result_text = Text(master, height=4, width=50)
result_text.grid(row=4, columnspan=2)

mainloop()


# In[ ]:




