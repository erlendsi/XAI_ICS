import pandas as pd
from sklearn.decomposition import PCA
from time import time
import numpy as np
import matplotlib.pyplot as plt; plt.style.use('dark_background')
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings('ignore')
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectPercentile, f_classif, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print('\n ---------------------------------- CICIoT2023 --------------------------------------------- \n')

DATASET_DIRECTORY = '../CICIoT2023/'
print(os.listdir(DATASET_DIRECTORY))

df_sets = [k for k in os.listdir(DATASET_DIRECTORY) if k.endswith('.csv')]
df_sets.sort()

# Initialize an empty DataFrame to store the concatenated data
concatenated_df = pd.DataFrame()

# Iterate through each CSV file and concatenate its contents to the DataFrame
for csv_file in tqdm(df_sets):
    file_path = os.path.join(DATASET_DIRECTORY, csv_file)
    df = pd.read_csv(file_path)
    concatenated_df = pd.concat([concatenated_df, df], ignore_index=True)

# Display the concatenated DataFrame
df = concatenated_df.copy()
del concatenated_df
df.dropna(inplace=True)
df.count()

X_columns = [
    'flow_duration', 'Header_Length', 'Protocol Type', 'Duration',
       'Rate', 'Srate', 'Drate', 'fin_flag_number', 'syn_flag_number',
       'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
       'ece_flag_number', 'cwr_flag_number', 'ack_count',
       'syn_count', 'fin_count', 'urg_count', 'rst_count', 
        'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP',
       'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min',
       'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Magnitue',
       'Radius', 'Covariance', 'Variance', 'Weight', 
]
y_column = 'label'

X_df = df[X_columns]
y_df = df[y_column]

y_df_binary = y_df.apply(lambda x: 'attack' if x != 'BenignTraffic' else x)
y_df_binary.value_counts()


scaler = MinMaxScaler()

scaled_df = X_df.copy()
for column in X_df.columns:
    column_data = X_df[column].values.reshape(-1, 1)
    scaled_df[column] = scaler.fit_transform(column_data)

X_train, X_test, y_train, y_test = train_test_split(scaled_df, y_df_binary, test_size=0.2, random_state=42)

print('\n ---------------------------------- BINARY CLASSIFICATION --------------------------------------------- \n')
#Training a classifier
clf = RandomForestClassifier(random_state = 69)
t0 = time()
clf.fit(X_train, y_train)
tt = time() - t0
print ("Classifier train: {}s".format(round(tt, 3)))

# Prediction on train set
t0 = time()
pred_train = clf.predict(X_train)
tt = time() - t0
print ("Train set prediction: {}s".format(round(tt, 3)))

# Prediction on test set
t0 = time()
pred_test = clf.predict(X_test)
tt = time() - t0
print ("Train set prediction: {}s".format(round(tt, 3)))

# Metrics
precision = precision_score(y_test, pred_test, average='micro')
recall = recall_score(y_test, pred_test, average='micro')
f1 = f1_score(y_test, pred_test, average='micro')
accuracy_train = accuracy_score(y_train, pred_train)
accuracy_test = accuracy_score(y_test, pred_test)

# Printing results
print("Accuracy train:", accuracy_train)
print("Accuracy test:", accuracy_test)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


print('\n ---------------------------------- NON - BINARY CLASSIFICATION --------------------------------------------- \n')

X_train, X_test, y_train, y_test = train_test_split(scaled_df, y_df, test_size=0.2, random_state=42)

#Training a classifier
clf = RandomForestClassifier(random_state = 69)
t0 = time()
clf.fit(X_train, y_train)
tt = time() - t0
print ("Classifier train: {}s".format(round(tt, 3)))

# Prediction on train set
t0 = time()
pred_train = clf.predict(X_train)
tt = time() - t0
print ("Train set prediction: {}s".format(round(tt, 3)))

# Prediction on test set
t0 = time()
pred_test = clf.predict(X_test)
tt = time() - t0
print ("Train set prediction: {}s".format(round(tt, 3)))

# Metrics
precision = precision_score(y_test, pred_test, average='micro')
recall = recall_score(y_test, pred_test, average='micro')
f1 = f1_score(y_test, pred_test, average='micro')
accuracy_train = accuracy_score(y_train, pred_train)
accuracy_test = accuracy_score(y_test, pred_test)

# Printing results
print("Accuracy train:", accuracy_train)
print("Accuracy test:", accuracy_test)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

print('\n ---------------------------------- Select n best features based on RF feature importance ---------------------------------- \n')

importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)

forest_importances = pd.Series(importances, index=scaled_df.columns)

# Create a DataFrame with feature names and importances
feature_importance_df = pd.DataFrame({'Feature': scaled_df.columns, 'Importance': importances})

# Sort the DataFrame by importance scores in descending order
feature_importance_df_sorted = feature_importance_df.sort_values(by='Importance', ascending=False)

def select_important_features(feature_importance_df_sorted, threshold=0.01):
    important_features = feature_importance_df_sorted[feature_importance_df_sorted['Importance'] > threshold]
    selected_feature_names = important_features['Feature'].tolist()
    return selected_feature_names

# Usage:
selected_features = select_important_features(feature_importance_df_sorted, threshold=0.01)
print("Selected features:", selected_features)

X_train, X_test, y_train, y_test = train_test_split(scaled_df[selected_features], y_df, test_size=0.2, random_state=42)

#Training a classifier
clf = RandomForestClassifier(random_state = 69)
t0 = time()
clf.fit(X_train, y_train)
tt = time() - t0
print ("Classifier train: {}s".format(round(tt, 3)))

# Prediction on train set
t0 = time()
pred_train = clf.predict(X_train)
tt = time() - t0
print ("Train set prediction: {}s".format(round(tt, 3)))

# Prediction on test set
t0 = time()
pred_test = clf.predict(X_test)
tt = time() - t0
print ("Train set prediction: {}s".format(round(tt, 3)))

# Metrics
precision = precision_score(y_test, pred_test, average='micro')
recall = recall_score(y_test, pred_test, average='micro')
f1 = f1_score(y_test, pred_test, average='micro')
accuracy_train = accuracy_score(y_train, pred_train)
accuracy_test = accuracy_score(y_test, pred_test)

# Printing results
print("Accuracy train:", accuracy_train)
print("Accuracy test:", accuracy_test)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("N best features:", len(selected_features))

print('\n ---------------------------------- PCA --------------------------------------------- \n')

pca = PCA().fit(scaled_df)

# Find the index where cumulative explained variance first exceeds 95%
index_95_percent = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95)

pca = PCA(n_components=index_95_percent)
pca_result = pca.fit_transform(scaled_df)

X_train, X_test, y_train, y_test = train_test_split(pca_result, y_df, test_size=0.2, random_state=42)

#Training a classifier
clf = RandomForestClassifier(random_state = 69)
t0 = time()
clf.fit(X_train, y_train)
tt = time() - t0
print ("Classifier train: {}s".format(round(tt, 3)))

# Prediction on train set
t0 = time()
pred_train = clf.predict(X_train)
tt = time() - t0
print ("Train set prediction: {}s".format(round(tt, 3)))

# Prediction on test set
t0 = time()
pred_test = clf.predict(X_test)
tt = time() - t0
print ("Train set prediction: {}s".format(round(tt, 3)))

# Metrics
precision = precision_score(y_test, pred_test, average='micro')
recall = recall_score(y_test, pred_test, average='micro')
f1 = f1_score(y_test, pred_test, average='micro')
accuracy_train = accuracy_score(y_train, pred_train)
accuracy_test = accuracy_score(y_test, pred_test)

# Printing results
print("Accuracy train:", accuracy_train)
print("Accuracy test:", accuracy_test)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Optimal number of components:", index_95_percent)