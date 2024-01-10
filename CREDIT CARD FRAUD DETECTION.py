#!/usr/bin/env python
# coding: utf-8

# In[92]:


pip install pandas scikit-learn imbalanced-learn


# In[93]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt


# In[94]:


# Load the dataset
data = pd.read_csv('creditcard.csv')


# In[95]:


pd.options.display.max_columns=None


# In[96]:


data


# # Preprocess

# In[97]:


data.isnull().sum()


# In[98]:


data.columns


# In[99]:


from sklearn.preprocessing import StandardScaler


# In[100]:


sc=StandardScaler()
data['Amount']=sc.fit_transform(pd.DataFrame(data['Amount']))


# In[101]:


data


# In[102]:


data


# In[103]:


data.duplicated().any()


# In[104]:


data=data.drop_duplicates()


# In[105]:


data.shape


# # Identify fraudulent credit card transactions

# In[118]:


classes = data['Class'].value_counts()
classes


# In[108]:


normal_share = round((classes[0]/data['Class'].count()*100),2)
normal_share


# In[109]:


fraud_share = round((classes[1]/data['Class'].count()*100),2)
fraud_share


# Only 0.17% frauds.

# In[110]:


#Barplot for number fraudulent vs non-fraudulent transcation
sns.countplot(x='Class', data=data)
plt.title('number fraudulent vs non-fraudulent transcation')
plt.show()


# In[111]:


# Bar plot for the percentage of fraudulent vs non-fraudulent transcations
fraud_percentage = {'Class':['Non-Fraudulent', 'Fraudulent'], 'Percentage':[normal_share, fraud_share]} 
data_fraud_percentage = pd.DataFrame(fraud_percentage) 
sns.barplot(x='Class',y='Percentage', data=data_fraud_percentage)
plt.title('Percentage of fraudulent vs non-fraudulent transcations')
plt.show()


# In[112]:


# Box plot for the distribution of fraudulent vs non-fraudulent transactions
sns.boxplot(x='Class', y='Amount', data=data, palette=['skyblue', 'lightcoral'])
plt.title('Distribution of Amount for Fraudulent vs Non-Fraudulent Transactions')
plt.show()


# In[114]:


fraud_counts_over_time = data.groupby(['Time', 'Class']).size().reset_index(name='Transaction Count')

# Line plot for the number of fraudulent vs non-fraudulent transactions over time
plt.figure(figsize=(12, 6))
sns.lineplot(x='Time', y='Transaction Count', data=fraud_counts_over_time, hue='Class', palette=['skyblue', 'lightcoral'])
plt.title('Number of Fraudulent vs Non-Fraudulent Transactions Over Time')
plt.xlabel('Time')
plt.ylabel('Number of Transactions')
plt.show()


# In[63]:


# Preprocess and normalize the data
X = data.drop('Class', axis=1)  # Features
y = data['Class']  # Target variable


# # Model training and testing

# In[64]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[65]:


# Normalize the data using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# # Handle class imbalance using oversampling (RandomOverSampler)

# In[66]:


# Handle class imbalance using oversampling (RandomOverSampler)
oversampler = RandomOverSampler(sampling_strategy='minority')
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)


# # Logistic Regression 

# In[67]:


# Train a classification algorithm (Logistic Regression in this case)
model = LogisticRegression()
model.fit(X_train_resampled, y_train_resampled)


# In[68]:


# Make predictions on the test set
y_pred = model.predict(X_test)


# In[69]:


# Evaluate the model's performance
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[70]:


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[71]:


accuracy_score(y_test, y_pred)


# # Evaluating the model's performance using metrics like precision

# In[72]:


precision_score(y_test, y_pred)


# # Evaluating the model's performance using metrics like recall

# In[73]:


recall_score(y_test,y_pred)


# # Evaluating the model's performance using metrics like F1-score

# In[74]:


f1_score(y_test,y_pred)


# In[ ]:


# The other three scores are less than accuracy because of imbalanced dataset so we haev handle imbalance dataset


# ##Handling imblanced dataset
# 
# Dealing with imbalanced datasets is a common challenge in machine learning, especially in classification problems where one class significantly outnumbers the other. Here are some strategies to handle imbalanced datasets:
# 
# 1. Resampling:
#    - Undersampling: Reduce the number of instances in the majority class to balance it with the minority class.
#    - Oversampling: Increase the number of instances in the minority class by duplicating or generating synthetic examples.
# 
# 2. Generating Synthetic Samples:
#    - Use techniques like SMOTE (Synthetic Minority Over-sampling Technique) to create synthetic instances of the minority class to balance the dataset.
# 
# 3. Change the Algorithm:
#    - Use algorithms that are less sensitive to class imbalance. For example, ensemble methods like Random Forests or XGBoost can handle imbalanced datasets better.
# 
# 

# In[76]:


# so we handled class imbalance using oversampling (RandomOverSampler)

