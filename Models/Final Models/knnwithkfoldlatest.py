# -*- coding: utf-8 -*-
"""KNNWithKFoldLatest.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1om6TcZ4KzdXq6QDab0iMdV058dMnMIc7
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from time import time
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('/scratch/vrames25/NIDS/datasets/csv_preprocessed.csv')

X = data.drop(columns=["Attack Type"])

Y = data["Attack Type"]

def mutual_info(X, Y):
  mutual_info_arr = mutual_info_classif(X, Y)
  series_info = pd.Series(mutual_info_arr)
  series_info.index = X.columns
  series_top = series_info.sort_values(ascending=False)[:25]
  return series_top

result = mutual_info(X, Y)

new_data = data[result.keys()]

# # Hard coding now to skip the mutual information step
# new_data = data[["SrcWin", "DstWin", "dHops", "dTtl", "TotBytes", "SrcBytes", "sMeanPktSz", "DstGap", "SrcGap", "dTos", "DstTCPBase", "SrcTCPBase", "TcpRtt", "Proto_udp", "DstBytes", "AckDat" , "dMeanPktSz", "Proto_tcp", "SynAck", "Load"]]
# new_data.head()

def concat_column_for_plot(pca_data, column_name):
  for_plot = pd.concat([pca_data, data[column_name]], axis = 1)
  return for_plot

new_data = concat_column_for_plot(new_data, "Attack Type")

X_train, X_test, y_train, y_test = train_test_split(new_data.loc[:, new_data.columns != 'Attack Type'], new_data['Attack Type'],
                                                    stratify=new_data['Attack Type'],
                                                    test_size=0.15)

X_train = pd.DataFrame(X_train, columns=new_data.columns.to_list()[:-1])
X_test = pd.DataFrame(X_test, columns=new_data.columns.to_list()[:-1])
y_train = pd.DataFrame(y_train, columns=['Attack Type'])
y_test = pd.DataFrame(y_test, columns=['Attack Type'])

print("Training dataset size:", X_train.shape)
print("Testing dataset size:", X_test.shape)
print("Training target size:", y_train.shape)
print("Testing target size:", y_test.shape)

def get_pca_df(scaled_data, no_of_components):
  pca = PCA(n_components=no_of_components)
  Principal_components=pca.fit_transform(scaled_data)
  column_names = ["PC "+str(i) for i in range(1, no_of_components+1)]
  pca_df = pd.DataFrame(data = Principal_components, columns = column_names)
  return pca_df, pca

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
Y = new_data['Attack Type']
X_train, pca = get_pca_df(X_train, 15)
start_time = time()
for train_index, test_index in skf.split(X_train, y_train):
    X1_train, X1_test = X_train.iloc[train_index], X_train.iloc[test_index]
    y1_train, y1_test = y_train.iloc[train_index], y_train.iloc[test_index]
    # Initialize kNN classifier
    knn = KNeighborsClassifier(n_neighbors=19)
    # Train the classifier
    knn.fit(X1_train, y1_train)
    # Predict on the test set
    y_pred = knn.predict(X1_test)
    # Calculate evaluation metrics and store them
    accuracy_scores.append(accuracy_score(y1_test, y_pred))
    precision_scores.append(precision_score(y1_test, y_pred, average='weighted'))
    recall_scores.append(recall_score(y1_test, y_pred, average='weighted'))
    f1_scores.append(f1_score(y1_test, y_pred, average='weighted'))

end_time = time()

print(f' Train start: {start_time}')
print(f' Train end: {end_time}')
print(f' Training time: {end_time - start_time} seconds\n')

test_start_time = time()

x_test = pca.transform(X_test)

print(x_test)

y1_pred = knn.predict(x_test)

print(accuracy_score(y_test, y1_pred))

print(accuracy_scores)

test_end_time = time()

print(f' Test start: {test_start_time}')
print(f' Test end: {test_end_time}')
print(f' Testing time: {test_end_time - test_start_time} seconds\n')

# Compute and print the confusion matrix
conf_matrix = confusion_matrix(y_test, y1_pred)
print('Confusion Matrix:')
print(conf_matrix)

le_name_mapping = {'Benign': 0, 'HTTPFlood': 1, 'ICMPFlood': 2, 'SYNFlood': 3, 'SYNScan': 4, 'SlowrateDoS': 5, 'TCPConnectScan': 6, 'UDPFlood': 7, 'UDPScan': 8}

# Creating  a confusion matrix,which compares the y_test and y_pred
cm = confusion_matrix(y_test, y1_pred)

# Creating a dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.
cm_df = pd.DataFrame(cm,
                     index = le_name_mapping.keys(),
                     columns = le_name_mapping.keys())

#Plotting the confusion matrix
plt.figure(figsize=(10,8))
sns.heatmap(cm_df, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.savefig('confusion_matrix_knn_multi.png')

precision = precision_score(y_test, y1_pred, average='weighted') # Use 'binary' for binary classification
print(f'Precision: {precision}')

recall = recall_score(y_test, y1_pred, average='weighted') # Use 'binary' for binary classification
print(f'Recall: {recall}')

f1 = f1_score(y_test, y1_pred, average='weighted') # Use 'binary' for binary classification
print(f'F1 Score: {f1}')

accuracy = accuracy_score(y_test, y1_pred)
print(f'Accuracy: {accuracy}')

# Assuming metrics are stored in these variables
metrics = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1,
    # 'ROC-AUC': roc_auc  # Uncomment if ROC-AUC is applicable and calculated
}

# Convert dictionary to lists for plotting
metric_names = list(metrics.keys())
metric_values = [metrics[metric] for metric in metric_names]

# Create bar plot
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(x=metric_values, y=metric_names, palette="viridis")

plt.xlabel('Score')
plt.ylabel('Metric')
plt.title('KNN Multi class Classification Metrics')
plt.xlim(0, 1)  # Assuming the scores are between 0 and 1
plt.savefig('knn_multi_class_metrics.png')