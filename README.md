import warnings
warnings.filterwarnings(action="ignore")

import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



#Loading the Red Wines dataset
df = pd.read_csv("winequality-red[1].csv" ,sep=';')

# Display first few records
df.head(10)

print(df.shape)

#checking for the null values in the dataset
print(df.isnull().sum())
print("\n\n")


#DATA ANALYSIS AND VASULIZATION

print(df.describe())
print("\n\n")

# number of values for each quality

sns.catplot(x='quality',data=df, kind='count')
print("\n\n")


# CORRELATION BETWEEN ALL THE COLUMN AND QUALITY

correlation=df.corr()
print('\n\n')

# constructing a heatmap to understand the correlation between the column

plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt= '.1f' , annot=True, annot_kws={'size':8} , cmap='Reds')
print('\n\n\n')


# data preprocessing

#separate the data and label

X=df.drop('quality',axis=1)
print(X)
print('\n\n')


# TRAIN AND TEST SPLIT

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)
print(Y.shape,Y_train.shape,Y_test.shape)

#training model

model=LogisticRegression()
model.fit(X_train,Y_train)

#Evaluation of the model

Y_pred=model.predict(X_test)
accuracy=accuracy_score(Y_test,Y_pred)
print(f"accuracy: {accuracy}")
print("\n\n")
print("confusion matrix:")
print(confusion_matrix(Y_test,Y_pred),'\n\n')
print("Classification Report:")
print(classification_report(Y_test,Y_pred))
