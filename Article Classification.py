import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import time
begin = time.time()
from sklearn.svm import SVC
from sklearn.metrics import r2_score,accuracy_score

dataframe = pd.read_csv("OnlineArticlesPopularity_Milestone2.csv")
pd.set_option("display.maX_columns",None)
dataframe

for i in dataframe.select_dtypes(include = "number").columns:
  sn.boxplot(data = dataframe,x = i)
  plt.show()
  
encoder = LabelEncoder()
dataframe["Article Popularity"] = encoder.fit_transform(dataframe["Article Popularity"])
article_popularity = dataframe["Article Popularity"]
dataframe.drop(columns=['Article Popularity'], inplace=True)


pt = PowerTransformer(method = "yeo-johnson")
numeric_columns = dataframe.select_dtypes(include = "number").columns
transformed_data = pt.fit_transform(dataframe[numeric_columns])
dataframe2 = pd.DataFrame(transformed_data, columns=numeric_columns, index=dataframe.index)


for column in dataframe2.columns:
  q1 = dataframe2[column].quantile(0.25)
  q3 = dataframe2[column].quantile(0.75)
  iqr = q3 - q1
  upper_limit = q3 + 1.5* iqr
  lower_limit = q1 - 1.5* iqr
  dataframe2.loc[dataframe2[column] > upper_limit,column] = upper_limit
  dataframe2.loc[dataframe2[column] < lower_limit,column] = lower_limit
dataframe2["Article Popularity"] = article_popularity

for i in dataframe2.select_dtypes(include = "number").columns:
  sn.boxplot(data = dataframe2,x = i)
  plt.show()
  
categorical_columns = dataframe.select_dtypes(include = "object").columns
dataframe3 = pd.DataFrame( dataframe[categorical_columns])

dataframe3.drop(columns=['url', 'title'], inplace=True)
for column in dataframe3.columns:
  if(column != 'channel type'):
    dataframe3[column] = encoder.fit_transform(dataframe3[column])
dataframe3 = pd.get_dummies(dataframe3, columns=['channel type'])

final_df = pd.DataFrame()
for column in dataframe2.columns:
  final_df[column] = dataframe2[column]
for column in dataframe3.columns:
  final_df[column] = dataframe3[column]
final_df

y = final_df["Article Popularity"]
x = final_df.drop(columns="Article Popularity")

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
models = {
    'Logistic Regression 1': LogisticRegression(max_iter=100),
    'Logistic Regression 2': LogisticRegression(max_iter=500),
    'Logistic Regression 3': LogisticRegression(max_iter=200),

    'Decision Tree': DecisionTreeClassifier(max_depth = 100),
    'Decision Tree': DecisionTreeClassifier(max_depth = 200),
    'Decision Tree': DecisionTreeClassifier(max_depth = 300),

    'Random Forest 1 ': RandomForestClassifier(n_estimators=50),
    'Random Forest 2 ': RandomForestClassifier(n_estimators=20),
    'Random Forest 3 ': RandomForestClassifier(n_estimators=10)
}

results = {}
training_times = {}
testing_times = {}

for model_name in models.keys():
    model = models[model_name]
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    start_time = time.time()
    predictions = model.predict(X_test)
    testing_time = time.time() - start_time

    accuracy = accuracy_score(predictions,y_test)
    results[model_name] = accuracy
    training_times[model_name] = training_time
    testing_times[model_name] = testing_time
