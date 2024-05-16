import pandas as pd
from sklearn.decomposition import PCA
from time import time
import numpy as np
import matplotlib.pyplot as plt; plt.style.use('dark_background')
import seaborn as sns
from tqdm import tqdm
import warnings; warnings.filterwarnings('ignore')
import psutil


from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectPercentile, f_classif, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print('\n ---------------------------------- KDD99 --------------------------------------------- \n')


#loading the data
col_names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot", 
             "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", 
             "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", 
             "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", 
             "srv_diff_host_rate", "dst_host_count","dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", 
             "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", 
             "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]

#Initially, we will use all features
num_features = ["duration", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", 
                "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
                "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", 
                "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
                "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", 
                "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                "dst_host_rerror_rate", "dst_host_srv_rerror_rate"]

kdd_df = pd.read_csv("../KDD99/corrected.csv", names = col_names)

features = kdd_df[num_features].astype(float)

print('\n ---------------------------------- BINARY CLASSIFICATION --------------------------------------------- \n')

#we are going to reduce the outputs to 'normal' and 'attack'
labels = kdd_df['label'].copy()
labels_binary = kdd_df['label'].copy()
labels_binary[labels_binary != 'normal.'] = 'attack.'

# Create a LabelEncoder instance
label_encoder = LabelEncoder()

encoded_labels_bin = labels_binary.copy()

# Fit the encoder on the labels and transform the column
encoded_labels_bin = label_encoder.fit_transform(encoded_labels_bin)

encoded_labels = labels.copy()

# Fit the encoder on the labels and transform the column
encoded_labels = label_encoder.fit_transform(encoded_labels)
# encoded_labels_series = pd.Series(encoded_labels)

# # Get the value counts
# value_counts = encoded_labels_series.value_counts()

# print(value_counts)

# Assuming 'features' is your DataFrame
scaler = MinMaxScaler()

# Apply Min-Max scaling to each column separately
features_scaled = features.copy()
for column in features.columns:
    column_data = features[column].values.reshape(-1, 1)
    features_scaled[column] = scaler.fit_transform(column_data)

# If you want to keep the scaled data in the original DataFrame
features_scaled.columns = features.columns


X_train, X_test, y_train, y_test = train_test_split(features_scaled, encoded_labels_bin, test_size=0.15, random_state=42)

#Training a classifier
clf = RandomForestClassifier(random_state = 69)
t0 = time()
clf.fit(X_train, y_train)
tt = time() - t0

# Prediction on train set
t0 = time()
pred_train = clf.predict(X_train)
tt = time() - t0

# Prediction on test set
t0 = time()
pred_test = clf.predict(X_test)
tt = time() - t0
print ("Test set prediction: {}s".format(round(tt, 3)))

# Metrics
precision = precision_score(y_test, pred_test)
recall = recall_score(y_test, pred_test)
f1 = f1_score(y_test, pred_test)
accuracy_train = accuracy_score(y_train, pred_train)
accuracy_test = accuracy_score(y_test, pred_test)

# Printing results
print("Accuracy train:", np.round(accuracy_train, 8))
print("Accuracy test:", np.round(accuracy_test, 8))
print("Precision:", np.round(precision, 8))
print("Recall:", np.round(recall, 8))
print("F1 Score:", np.round(f1, 8))

print('\n ---------------------------------- NON - BINARY CLASSIFICATION --------------------------------------------- \n')

X_train, X_test, y_train, y_test = train_test_split(features_scaled, encoded_labels, test_size=0.15, random_state=42)

#Training a classifier
clf = RandomForestClassifier(random_state = 69)
t0 = time()
clf.fit(X_train, y_train)
tt = time() - t0

# Prediction on train set
t0 = time()
pred_train = clf.predict(X_train)
tt = time() - t0

# Prediction on test set
t0 = time()
pred_test = clf.predict(X_test)
tt = time() - t0
print ("Train set prediction: {}s".format(round(tt, 3)))

# Metrics
precision = precision_score(y_test, pred_test, average='weighted')
recall = recall_score(y_test, pred_test, average='weighted')
f1 = f1_score(y_test, pred_test, average='weighted')
accuracy_train = accuracy_score(y_train, pred_train)
accuracy_test = accuracy_score(y_test, pred_test)

# Printing results
print("Accuracy train:", np.round(accuracy_train, 8))
print("Accuracy test:", np.round(accuracy_test, 8))
print("Precision:", np.round(precision, 8))
print("Recall:", np.round(recall, 8))
print("F1 Score:", np.round(f1, 8))

print('\n ---------------------------------- Select n best features based on RF feature importance ---------------------------------- \n')

importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)

forest_importances = pd.Series(importances, index=features.columns)

# Create a DataFrame with feature names and importances
feature_importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': importances})

# Sort the DataFrame by importance scores in descending order
feature_importance_df_sorted = feature_importance_df.sort_values(by='Importance', ascending=False)

def select_important_features(feature_importance_df_sorted, threshold=0.01):
    important_features = feature_importance_df_sorted[feature_importance_df_sorted['Importance'] > threshold]
    selected_feature_names = important_features['Feature'].tolist()
    return selected_feature_names

# Usage:
selected_features = select_important_features(feature_importance_df_sorted, threshold=0.01)
print("Selected features:", selected_features)

X_train, X_test, y_train, y_test = train_test_split(features_scaled[selected_features], encoded_labels, test_size=0.15, random_state=42)

#Training a classifier
clf = RandomForestClassifier(random_state = 69)
t0 = time()
clf.fit(X_train, y_train)
tt = time() - t0

# Prediction on train set
t0 = time()
pred_train = clf.predict(X_train)
tt = time() - t0

# Prediction on test set
t0 = time()
pred_test = clf.predict(X_test)
tt = time() - t0
print ("Test set prediction: {}s".format(round(tt, 3)))

# Metrics
precision = precision_score(y_test, pred_test, average='weighted')
recall = recall_score(y_test, pred_test, average='weighted')
f1 = f1_score(y_test, pred_test, average='weighted')
accuracy_train = accuracy_score(y_train, pred_train)
accuracy_test = accuracy_score(y_test, pred_test)

# Printing results
print("Accuracy train:", np.round(accuracy_train, 8))
print("Accuracy test:", np.round(accuracy_test, 8))
print("Precision:", np.round(precision, 8))
print("Recall:", np.round(recall, 8))
print("F1 Score:", np.round(f1, 8))
print("N best features:", len(selected_features))

print('\n ---------------------------------- PCA --------------------------------------------- \n')

pca = PCA().fit(features_scaled)
# Find the index where cumulative explained variance first exceeds 95%
index_99_percent = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95)
index_99_percent = index_99_percent + 1

pca = PCA(n_components=index_99_percent)
pca_result = pca.fit_transform(features_scaled)

X_train, X_test, y_train, y_test = train_test_split(pca_result, encoded_labels, test_size=0.15, random_state=42)

#Training a classifier
clf = RandomForestClassifier(random_state = 69)
t0 = time()
clf.fit(X_train, y_train)
tt = time() - t0

# Prediction on train set
t0 = time()
pred_train = clf.predict(X_train)
tt = time() - t0

# Prediction on test set
t0 = time()
pred_test = clf.predict(X_test)
tt = time() - t0
print ("Train set prediction: {}s".format(round(tt, 3)))

# Metrics
precision = precision_score(y_test, pred_test, average='weighted')
recall = recall_score(y_test, pred_test, average='weighted')
f1 = f1_score(y_test, pred_test, average='weighted')
accuracy_train = accuracy_score(y_train, pred_train)
accuracy_test = accuracy_score(y_test, pred_test)

# Printing results
print("Accuracy train:", np.round(accuracy_train, 8))
print("Accuracy test:", np.round(accuracy_test, 8))
print("Precision:", np.round(precision, 8))
print("Recall:", np.round(recall, 8))
print("F1 Score:", np.round(f1, 8))
print("Number of components:", index_99_percent)