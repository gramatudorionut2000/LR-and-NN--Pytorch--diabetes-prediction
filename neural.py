import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import tqdm
from sklearn.model_selection import StratifiedKFold
data1=pd.read_csv('dataset1.csv')
data2=pd.read_csv('dataset2.csv')
data3=pd.read_csv('dataset3.csv')
data3=data3.drop('Id',axis=1)
data=pd.concat([data1,data2,data3]).drop_duplicates()
data_missing_values=data.copy()
Labels=['Missing values Unchanged', 'Missing Values replaced with the mean', 'Missing values replaced with the median']
cols=['Glucose', 'BloodPressure', 'SkinThickness', 'BMI']


def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc=roc_auc_score(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred)
    return accuracy, precision, recall, f1, auc, confusion
labels=['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'confusion\n']

for column in cols:
    data[column].replace(0, np.nan, inplace=True)
df_no_diabetes=data.loc[data['Outcome'] == 0]
df_diabetes=data.loc[data['Outcome'] == 1]
no_diabetes=df_no_diabetes.copy()
with_diabetes=df_diabetes.copy()
for column in cols:
    no_diabetes[column]=no_diabetes[column].fillna(no_diabetes[column].mean(skipna=True))
    with_diabetes[column]=with_diabetes[column].fillna(with_diabetes[column].mean(skipna=True))
data_mean=pd.concat([no_diabetes,with_diabetes])
data_mean = data_mean.sample(frac = 1)
no_diabetes=df_no_diabetes.copy()
with_diabetes=df_diabetes.copy()
for column in cols:
    no_diabetes[column]=no_diabetes[column].fillna(no_diabetes[column].median(skipna=True))
    with_diabetes[column]=with_diabetes[column].fillna(with_diabetes[column].median(skipna=True))
data_median=pd.concat([no_diabetes,with_diabetes])
data_median = data_median.sample(frac = 1)
datasets=[data_missing_values, data_mean, data_median]


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

def model_train(x_train, y_train, x_test, y_test):
    NN=nn.Sequential(
    nn.Linear(8, 12),
    nn.LeakyReLU(),
    nn.Linear(12, 12),
    nn.LeakyReLU(),
    nn.Linear(12, 8),
    nn.LeakyReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid())

    loss_function = nn.BCELoss()
    optimizer = torch.optim.AdamW(NN.parameters(), lr=0.01)

    epochs = 200   
    batch_size = 16 
    batches_per_epoch = len(x_train) // batch_size

    for epoch in range(epochs):
        with tqdm.trange(batches_per_epoch, disable=True) as bar:
            for i in bar:

                start = i * batch_size
                x_batch = x_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]

                y_pred = NN(x_batch)
                loss = loss_function(y_pred, y_batch)

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()


    y_pred = NN(x_test)
    return calculate_metrics(y_test, y_pred.detach().numpy().round())


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for dataset, label in zip(datasets,Labels):
    print(f"---------{label}---------\n")
    predictors = data.select_dtypes(include=['int64', 'float64']).columns.tolist()[:-1]
    x= dataset[predictors].values
    y = dataset['Outcome'].values 
    best_scores=dict()
    mean_scores=dict()
    for index,metric in enumerate(labels):
        if index<len(labels)-1:
            best_scores[f'{metric}']=float('-inf')
            best_scores[f'{metric} index']=float('-inf')
            mean_scores[f'Mean {metric}']=0
    for iter, (train, test) in enumerate(kfold.split(x, y)):
        
        x_train, x_test = x[train], x[test]
        y_train, y_test = y[train], y[test]
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        x_test = torch.tensor(x_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
        metrics = model_train(x_train, y_train, x_test, y_test)
        print(f'FOLD {iter+1}')
        for index, (metric, label) in enumerate(zip(metrics, labels)):
            print(f'{label}:{metric}')
            if index <len(metrics)-1 and metric > best_scores[f'{label}']:
                best_scores[f'{label}']= metric
                best_scores[f'{label} index']= iter+1
            if index<len(metrics)-1:
                mean_scores[f'Mean {label}']=mean_scores[f'Mean {label}']+ metric
    for key, value in best_scores.items():
        print(f'{key}: {value}')
    for key, value in mean_scores.items():
        print(f'{key}: {value/(iter+1)}')

