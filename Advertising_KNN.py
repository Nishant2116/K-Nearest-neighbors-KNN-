#!/usr/bin/env python
# coding: utf-8

# # Advertising Dataset
# ### This Dataset contain the Information about the users either they clicked on Advertisement or not. Depending on the features like Daily Time spent on a site, Age, Area Income,etc.

# ## Importing Libraries

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings 
warnings.filterwarnings(action='ignore')


# ## Importing the Dataset

# In[2]:


df=pd.read_csv('advertising.csv')


# In[29]:


df.head(3)


# ## Exploratory Data analysis (EDA)

# > Area Income vs Age (hue=Sex)

# In[79]:


sns.set_style('whitegrid')
sns.jointplot(df['Age'],df['Area Income'],hue=df['Male'],palette='Set1');


# > Area Income (histogram)

# In[75]:


plt.figure(figsize=(15,4),dpi=300)
plt.hist(df['Area Income'],color='green')
plt.xlabel('Area Income');


# > Pairplot (Dataset)

# In[78]:


sns.pairplot(df,hue='Male');


# > Correlation of all Features (Heatmap)

# In[90]:


plt.figure(figsize=(17,8),dpi=300)
sns.heatmap(df.corr(),fmt='g',annot=True);


# ## Checking for the null values

# > Heatmap to check null values

# In[83]:


plt.figure(figsize=(17,4),dpi=300)
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis');


# ## Adjusting the data 

# In[5]:


df.head()


# In[6]:


df.drop((['Ad Topic Line','Country','Timestamp','City']),axis=1,inplace=True)


# In[7]:


df


# ## Scaling the Data

# In[8]:


from sklearn.preprocessing import StandardScaler


# In[9]:


scaler=StandardScaler()


# In[10]:


scaler.fit(df.drop('Clicked on Ad',axis=1))


# In[11]:


scaler_features=scaler.transform(df.drop('Clicked on Ad',axis=1))


# In[12]:


df_fit=pd.DataFrame(scaler_features,columns=df.columns[:-1])


# In[13]:


df_fit


# ### 0 null values : ready to deal with the model 

# ## Train Test Split

# In[14]:


df


# In[15]:


X=df_fit
y=df['Clicked on Ad']


# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# ## Creating the model

# In[18]:


from sklearn.neighbors import KNeighborsClassifier


# In[19]:


knn=KNeighborsClassifier(n_neighbors=1)


# In[20]:


knn.fit(X_train,y_train)


# ## Predictions & Evaluations

# In[21]:


predictions= knn.predict(X_test)


# In[22]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[23]:


print(classification_report(y_test,predictions))
print('\n')
print(confusion_matrix(y_test,predictions))
print('\n')
print("Acccuracy : ",accuracy_score(y_test,predictions)*100,'%')


# ## Finding the best k value for model 

# In[24]:


error_rate=[]
for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i=knn.predict(X_test)
    error_rate.append(np.mean(pred_i!=y_test))


# In[27]:


sns.set_style('whitegrid')
plt.figure(figsize=(20,6),dpi=300)
plt.plot(range(1,40),error_rate,color='blue',marker='o',ls='dashed',markerfacecolor='red',markersize=15)
plt .title('Error rate VS K value')
plt.xlabel('K')
plt.ylabel('Error rate');


# #### According to the plot k>20 gives maximum accuracy

# > for k==24,31,etc. we can get the maximum accuracy.

# ## Training the model for k==31

# In[28]:


knn=KNeighborsClassifier(n_neighbors=31)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
print('Classfication report: \n\n',classification_report(y_test,pred))
print('\n')
print('Confusion matrix : \n\n',confusion_matrix(y_test,pred))
print('\n')
print('Accuracy :',accuracy_score(y_test,pred)*100,'%')


# ## Testing the model 
# ### To test the model have to pass all the list of values require to train the X of model, but due to KNN we have to get values from user & have to scale them with dataset to get maximum accuracy through KNN model

# > Taking values from user

# In[ ]:


y_test


# In[ ]:


df.head()


# In[ ]:


Daily_Time_Spent_on_Site=float(input('Daily Time Spent on site :'))
Daily_Time_Spent_on_Site=float(input('Income :'))
Daily_Internet_Usage=float(input('Daily internet Usage :'))
Male=float(input('0=Female / 1=Male ?'))


# In[ ]:


arr=[[Daily_Time_Spent_on_Site,Daily_Time_Spent_on_Site,Daily_Internet_Usage,Male]]


# In[ ]:


scaler.fit(arr)


# In[ ]:


scaler_features=scaler.transform(arr)


# In[ ]:


scaler_features


# In[ ]:


X_test


# In[ ]:


df_user=pd.DataFrame(arr,columns=['Daily_Time_Spent_on_Site','Daily_Time_Spent_on_Site','Daily_Internet_Usage','Male'])


# In[ ]:


df_user


# In[ ]:


predictions=knn.predict(df_user)
if predictions[0]==0:
    print(em.emojize(':green_circle:'),"Employee is not going to leave the company.",em.emojize(':green_circle:'))
else:
    print(em.emojize(':prohibited:'),'Employee is going to leave the comapany',em.emojize(':prohibited:'))
    
    

