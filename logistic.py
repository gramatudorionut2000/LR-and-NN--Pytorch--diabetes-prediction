import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from matplotlib import pyplot as plt


data1=pd.read_csv('dataset1.csv')
data2=pd.read_csv('dataset2.csv')
data3=pd.read_csv('dataset3.csv')
data3=data3.drop('Id',axis=1)
data=pd.concat([data1,data2])
data=pd.concat([data,data3])
data=data.drop_duplicates()
cols=['Glucose', 'BloodPressure', 'SkinThickness', 'BMI']
data_missing_values=data.copy()
for column in cols:
    data[column].replace(0, np.nan, inplace=True)
df_no_diabetes=data.loc[data['Outcome'] == 0]
df_diabetes=data.loc[data['Outcome'] == 1]
no_diabetes=df_no_diabetes.copy()
with_diabetes=df_diabetes.copy()

for column in cols:
    print(no_diabetes[column].mean())
    no_diabetes[column]=no_diabetes[column].fillna(no_diabetes[column].mean(skipna=True))
    print(with_diabetes[column].mean())
    with_diabetes[column]=with_diabetes[column].fillna(with_diabetes[column].mean(skipna=True))
data_mean=pd.concat([no_diabetes,with_diabetes],ignore_index=True)
no_diabetes=df_no_diabetes.copy()
with_diabetes=df_diabetes.copy()
for column in cols:
    no_diabetes[column]=no_diabetes[column].fillna(no_diabetes[column].median(skipna=True))
    with_diabetes[column]=with_diabetes[column].fillna(with_diabetes[column].median(skipna=True))
data_median=pd.concat([no_diabetes,with_diabetes],ignore_index=True)
datasets=[data_missing_values, data_median, data_mean]
Labels=['Missing values Unchanged', 'Missing Values replaced with the mean', 'Missing values replaced with the median']
corr = data.corr()
sns.heatmap(corr, cmap="crest", annot=True)
plt.show()
for data, label in zip(datasets,Labels):
    print(f"---------{label}---------\n")
    predictors = data.select_dtypes(include=['int64', 'float64']).columns.tolist()[:-1]
    x= data[predictors]
    y = data['Outcome']
    model = LogisticRegression(solver='lbfgs', max_iter=500, random_state=42)
    metrics=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    scores = cross_validate(model, x, y, cv=10, scoring=metrics)
    best_scores=dict()
    mean_scores=dict()
    for metric in metrics:
        best_scores[f'{metric}']=float('-inf')
        best_scores[f'{metric} index']=float('-inf')
        mean_scores[f'Mean {metric}']=0
    for metric in metrics:
        total=0
        for index,value in enumerate(scores[f'test_{metric}']):
            total=total + value
            if value > best_scores[f'{metric}']:
                best_scores[f'{metric}']= value
                best_scores[f'{metric} index']= index
        mean_scores[f'Mean {metric}']=(total/10)
    for key, value in best_scores.items():
        print(f'{key}: {value}')
    for key, value in mean_scores.items():
        print(f'{key}: {value}')
