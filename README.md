
## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices.
5.Display the results.

## Program:
```
## Developed by: RAGHUL V
## RegisterNumber: 212223240132

import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])


```

## Output:
![image](https://github.com/Rahulv2005/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/152600335/c1d0f9cc-596c-4717-9070-20f2d8439f8d)

![image](https://github.com/Rahulv2005/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/152600335/7c3827b6-5305-4f9d-b77f-d2d76bd2b12d)

![image](https://github.com/Rahulv2005/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/152600335/99ceae61-f189-409d-8e9d-d06cf92f1a9a)

![image](https://github.com/Rahulv2005/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/152600335/618930e8-454c-4dad-9973-7ae15a272540)

![image](https://github.com/Rahulv2005/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/152600335/2903dbf2-c68c-4a9e-8924-d1c5eaaf1933)


![image](https://github.com/Rahulv2005/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/152600335/13876f87-f54e-4d56-a238-f1876d8a5466)

![image](https://github.com/Rahulv2005/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/152600335/ee5f7727-b074-4799-b8ef-47d3b05404e8)

![image](https://github.com/Rahulv2005/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/152600335/a27d3d5e-da31-4340-9a3d-0c0ec53104b4)


![image](https://github.com/Rahulv2005/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/152600335/8c80b3b9-54a6-4f61-a9ce-a6813bf533e1)

![image](https://github.com/Rahulv2005/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/152600335/5c67cd7a-2cbc-40e0-8275-0208098b9bed)

![image](https://github.com/Rahulv2005/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/152600335/93e4d26e-5e0e-40d0-b6a3-1d66ce7a83c9)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
