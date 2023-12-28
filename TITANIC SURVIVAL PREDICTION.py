#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pandas scikit-learn matplotlib


# # Step 1: Import Libraries and Load Data

# In[21]:


import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression


# In[3]:


get_ipython().run_line_magic('ls', '')


# In[4]:


# Load the Titanic dataset
data = pd.read_csv('Titanic-Dataset.csv')


# In[5]:


data


# # Step 2: Explore and Preprocess Data

# In[6]:


# Explore the dataset
print(data.info())


# In[10]:


print(data.describe())


# In[7]:


print(data.isnull().sum())


# In[8]:


# Handle missing values
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data['Cabin'].fillna(data['Cabin'].mode()[0], inplace=True)


# In[9]:


print(data.isnull().sum())


# In[10]:


# Convert categorical variables into numerical
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})


# In[11]:


data


# In[20]:


data.columns


# In[12]:


# Select relevant features for training
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = data[features]
y = data['Survived']


# # Step 3: Split the Data into Training and Testing Sets

# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Step 4: Build and Train the Model

# In[14]:


# Initialize the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)


# In[15]:


# Train the model
model.fit(X_train, y_train)


# # Step 5: Make Predictions and Evaluate the Model

# In[16]:


# Make predictions on the test set
y_pred = model.predict(X_test)


# In[37]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')


# # Step 6: Visualize the Results

# In[18]:


# Visualize feature importance
feature_importance = model.feature_importances_
plt.barh(features, feature_importance)
plt.xlabel('Feature Importance')
plt.title('Random Forest Classifier - Feature Importance')
plt.show()


# 1. feature_importance = model.feature_importances_:
#    - This line extracts the feature importance scores from the trained Random Forest model (`model`).
#    - In a Random Forest model, each feature is assigned an importance score, which indicates how much that feature contributes to the model's predictions.
# 
# 2. plt.barh(features, feature_importance):
#    - This line creates a horizontal bar chart using the `matplotlib` library.
#    - `features` is a list containing the names of the features used in the model (e.g., 'Pclass', 'Sex', 'Age', etc.).
#    - `feature_importance` is the list of importance scores corresponding to each feature.
# 

# In[22]:


# Distribution of survival
sns.countplot(x='Survived', data=data)
plt.title('Distribution of Survival')
plt.show()


# In[23]:


# Survival distribution by Pclass
sns.countplot(x='Pclass', hue='Survived', data=data)
plt.title('Survival Distribution by Pclass')
plt.show()


# In[24]:


# Age distribution by survival
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='Age', hue='Survived', kde=True, bins=30)
plt.title('Age Distribution by Survival')
plt.show()


# In[25]:


# Fare distribution by survival
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='Fare', hue='Survived', kde=True, bins=30)
plt.title('Fare Distribution by Survival')
plt.show()


# In[26]:


# Pairplot for numerical features
num_features = ['Age', 'SibSp', 'Parch', 'Fare']
sns.pairplot(data=data, hue='Survived', vars=num_features)
plt.suptitle('Pairplot for Numerical Features by Survival', y=1.02)
plt.show()


# In[27]:


# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# # Random Forest Classifier

# In[40]:


rf_model = RandomForestClassifier(random_state=42)

# Parameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)


# In[29]:


# Get the best parameters from the grid search
best_params_rf = grid_search_rf.best_params_


# In[30]:


best_params_rf 


# In[31]:


# Train the Random Forest model with the best parameters
best_rf_model = RandomForestClassifier(random_state=42, **best_params_rf)
best_rf_model.fit(X_train, y_train)


# In[33]:


# Logistic Regression
lr_model = LogisticRegression(random_state=42, max_iter=1000)

# Train the Logistic Regression model
lr_model.fit(X_train, y_train)

# Make predictions on the test set for both models
y_pred_rf = best_rf_model.predict(X_test)
y_pred_lr = lr_model.predict(X_test)


# In[34]:


y_pred_rf


# In[35]:


y_pred_lr


# In[38]:


# Evaluate the Logistic Regression model
accuracy_lr = accuracy_score(y_test, y_pred_lr)
conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)
classification_rep_lr = classification_report(y_test, y_pred_lr)

print("\nLogistic Regression:")
print(f'Accuracy: {accuracy_lr}')
print(f'Confusion Matrix:\n{conf_matrix_lr}')
print(f'Classification Report:\n{classification_rep_lr}')


# # Visualizing feature importance for Random Forest:

# In[39]:


# Visualize feature importance for Random Forest
feature_importance_rf = best_rf_model.feature_importances_
plt.barh(features, feature_importance_rf)
plt.xlabel('Feature Importance (Random Forest)')
plt.title('Random Forest Classifier - Feature Importance')
plt.show()


# In[ ]:




