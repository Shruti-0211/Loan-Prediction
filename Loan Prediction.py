#!/usr/bin/env python
# coding: utf-8

# # CodeClause Internship Program (January 2023)
# **Author: Yeole Akash Rajesh**

# In[2]:


import numpy as np
import pandas as pd


# # Features of our data
# 

# 1.LoanID= Unique Loan ID
# 
# 2.Gender= Male/ Female
# 
# 3.Married= Applicant married (Y/N)
# 
# 4.Dependents= Number of dependents
# 
# 5.Education= Applicant Education (Graduate/ Under Graduate)
# 
# 6.SelfEmployed= Self-employed (Y/N)
# 
# 7.ApplicantIncome= Applicant income
# 
# 8.CoapplicantIncome= Coapplicant income
# 
# 9.LoanAmount= Loan amount in thousands
# 
# 10.LoanAmountTerm= Term of the loan in months
# 
# 11.CreditHistory= Credit history meets guidelines 12.PropertyArea= Urban/ Semi-Urban/ Rural
# 
# 13.LoanStatus= (Target) Loan approved (Y/N)

# # Importing Libraries

# In[3]:


import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[4]:


df=pd.read_csv("D:\\Internships\\CodeClause\\dataset\\train_u6lujuX_CVtuZ9i.csv")
df.head()


# **Checking test data**
# 

# In[5]:


test=pd.read_csv("D:\\Internships\\CodeClause\\dataset\\test_Y3wMUE5_7gLdaTN.csv")
test.head()


# In[6]:


df.info()


# # Handling null values

# **Replacing Null Value by Mode**

# In[7]:


print('Gender Mode: ',df['Gender'].mode())
print('Married mode: ',df['Married'].mode())
print('Self_Employed',df['Self_Employed'].mode())
print('Credit_History',df['Credit_History'].mode())


# In[8]:


df[['Loan_Amount_Term','LoanAmount']][df['Loan_Amount_Term'].isnull()]


# In[9]:


df['Dependents'].value_counts()


# In[10]:


df['Dependents'].replace('3+',3,inplace=True)
df['Dependents'].value_counts()


# In[11]:


df[['Dependents','Married']][df['Dependents'].isnull()]


# **Replacing null values with mean for numberic and mode for object data type**

# In[12]:


df['Gender'].fillna('Male',inplace=True)#replacing with mode
df['Married'].fillna('Yes',inplace=True)#replacing with mode
df['Self_Employed'].fillna('No',inplace=True)#replacing with mode
df['LoanAmount'].fillna((df['LoanAmount'].mean()),inplace=True)#replacing with mean
df['Loan_Amount_Term'].fillna(84,inplace=True)#replacing with suitable option after visual analysis
df['Credit_History'].fillna(1.0,inplace=True)#replacing with mode
df['Dependents'].fillna(0,inplace=True)#replacing with mode


# In[13]:


df['Dependents']=df['Dependents'].astype('int')
df['Dependents'].dtype


# In[14]:


df.isnull().sum()


# **Missing values has now been handled**

# **Checking unique values in our dataset for better understanding**

# In[15]:


df.nunique()


# **Checking the description of our data to check for skewness and distribution**

# In[16]:


df.describe()


# # Graphical Visualiztion

# In[17]:


fig, axs = plt.subplots(figsize=(25,6),ncols=6,nrows=2)
sns.countplot(x=df['Loan_Status'],ax=axs[0,0])

sns.countplot(x=df['Gender'],hue=df['Loan_Status'],ax=axs[0,1])

sns.countplot(x=df['Married'],hue=df['Loan_Status'],ax=axs[0,2])

sns.countplot(x=df['Dependents'],hue=df['Loan_Status'],ax=axs[0,3])

sns.countplot(x=df['Education'],hue=df['Loan_Status'],ax=axs[0,4])

sns.countplot(x=df['Self_Employed'],hue=df['Loan_Status'],ax=axs[0,5])

sns.countplot(x=df['Credit_History'],hue=df['Loan_Status'],ax=axs[1,0])

sns.countplot(x=df['Property_Area'],hue=df['Loan_Status'],ax=axs[1,1])

sns.countplot(x=df['Gender'],hue=df['Dependents'],ax=axs[1,2])

sns.countplot(x=df['Loan_Amount_Term'],hue=df['Loan_Status'],ax=axs[1,3])

sns.countplot(x=df['Married'],hue=df['Dependents'],ax=axs[1,4])

sns.countplot(x=df['Education'],hue=df['Self_Employed'],ax=axs[1,5])
plt.show()


# **Interpretation**
# 
# 1.Loan_Status is our target variable which seems to be imbalanced as its in 1:3 ratio which needs to be worked on for our model to make accurate predictions.
# 
# 2.Males are more likely to be eligible for loans compared to female.
# 
# 3.Married are more likely to be eligible for loans compared to non married.
# 
# 4.People with less dependents are eligible for loans.
# 
# 5.Graduates are eligible for loan compared to non graduate.
# 
# 6.Non Selfemployed are eligible for loan compared to selfemployed.
# 
# 7.Poeple with good credit score are more likely to be eligible for loan compared to low credit scored.
# 
# 8.Semiurban and Urban people have are eligible for loan compared to rural population.
# 
# 9.Male have more dependents compared to female.
# 
# 10.Mostly people opt for 360 months term loan.
# 
# 11.Married people have more dependents compared to female.
# 
# 12.Graduates are selfemployed compared to non graduates.

# # Correlation of data

# In[18]:


plt.figure(figsize=(15,5))
sns.heatmap(df.corr(),annot=True)
plt.show()


# **Interpretation**
# 
# 1.Credit history is highly correlated to our target
# 
# 2.Education, self_employed, coapplicant income, loan amount, applicant income has negative correlation
# 
# 3.Loan amount and applicant are highly correlated
# 
# 4.Gender-Married & Dependents-Married are correlated
# 
# 5.This reflects that there are multicoreniality

# ***Fetching all object data typecolumns to covert***
# 

# In[35]:


obj_col=df.select_dtypes('object').columns
obj_col


# # Converting object to numeric data type
# 

# In[36]:


from sklearn.preprocessing import OrdinalEncoder
oe=OrdinalEncoder()
df[obj_col]=oe.fit_transform(df[obj_col])
df.head(3)


# In[37]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
df.iloc[:,:-1]=ss.fit_transform(df.iloc[:,:-1])
df.head()


# # Splitting dataset
# 

# In[38]:


x=df.iloc[:,:-1]
y=df.iloc[:,-1]
x.head()


# In[39]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=4,test_size=0.25,stratify=y)


# # Creating model function to test multiple models and choose the ideal one

# In[40]:


def mymodel(model):
    model.fit(xtrain,ytrain)
    ypred=model.predict(xtest)
    train_accuracy=model.score(xtrain,ytrain)
    test_accuracy=model.score(xtest,ytest)
    print(str(model)[:-2],'Accuracy')
    print('Accuracy:',accuracy_score(ytest,ypred),"\nClassification Report:\n",classification_report(ytest,ypred),           '\nConfusion Matrix: \n', confusion_matrix(ytest,ypred))
    print(f'Training Accuracy: {train_accuracy}\nTesting Accuracy :{test_accuracy}')
    print()
    print()
    return model


# # Testing the accuracy of our model
# 

# In[42]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

knn=mymodel(KNeighborsClassifier())
svc=mymodel(SVC())
dt=mymodel(DecisionTreeClassifier())
lr=mymodel(LogisticRegression())
gnb=mymodel(GaussianNB())
rfc=mymodel(RandomForestClassifier(n_estimators=80,max_depth=10,min_samples_leaf=12))


# # Conclusion:
# **KNN, Logistic regression,GaussianNB, and Random forest classifier...all seem to give the best accuracy of 82% with 0 false positive errors but our recall and precision doesnt seem to be upto mark so lets try imb-learn to balance our target variable ad train our model**

# In[ ]:




