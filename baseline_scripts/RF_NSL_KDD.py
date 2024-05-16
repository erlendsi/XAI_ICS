import pandas as pd
from sklearn.decomposition import PCA
from time import time
import numpy as np
import matplotlib.pyplot as plt; plt.style.use('defa')
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectPercentile, f_classif, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print('\n ---------------------------------- NSL-KDD --------------------------------------------- \n')

col_names = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot'
            ,'num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations'
            ,'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate'
            ,'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count'
            ,'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate'
            ,'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate'
            ,'outcome','level']

num_features = ['duration','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot'
            ,'num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations'
            ,'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate'
            ,'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count'
            ,'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate'
            ,'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate'
            ,'level']

kdd_train = pd.read_csv("../NSL_KDD/KDDTrain+.txt", names = col_names)
kdd_test = pd.read_csv("../NSL_KDD/KDDTest+.txt", names = col_names)

nsl_kdd_train = kdd_train.copy()
nsl_kdd_test = kdd_test.copy()

print('\n ---------------------------------- BINARY CLASSIFICATION --------------------------------------------- \n')


X_train = nsl_kdd_train[num_features]
X_test = nsl_kdd_test[num_features]
y_train = nsl_kdd_train["outcome"]
y_test = nsl_kdd_test["outcome"]

y_train = y_train.apply(lambda x: 'attack' if x != 'normal' else x)
y_test = y_test.apply(lambda x: 'attack' if x != 'normal' else x)

# Create a LabelEncoder instance
label_encoder = LabelEncoder()

encoded_y_train = y_train.copy()

# Fit the encoder on the labels and transform the column
encoded_y_train = label_encoder.fit_transform(encoded_y_train)

encoded_y_test = y_test.copy()

# Fit the encoder on the labels and transform the column
encoded_y_test = label_encoder.fit_transform(encoded_y_test)

# Assuming 'features' is your DataFrame
scaler = MinMaxScaler()

# Apply Min-Max scaling to each column separately
X_train_scaled = X_train.copy()
for column in X_train.columns:
    column_data = X_train[column].values.reshape(-1, 1)
    X_train_scaled[column] = scaler.fit_transform(column_data)

# If you want to keep the scaled data in the original DataFrame
X_train_scaled.columns = X_train.columns

# Apply Min-Max scaling to each column separately
X_test_scaled = X_test.copy()
for column in X_test.columns:
    column_data = X_test[column].values.reshape(-1, 1)
    X_test_scaled[column] = scaler.fit_transform(column_data)

# If you want to keep the scaled data in the original DataFrame
X_test_scaled.columns = X_test.columns

#Training a classifier
clf = RandomForestClassifier(random_state = 69)
t0 = time()
clf.fit(X_train, encoded_y_train)
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
precision = precision_score(encoded_y_test, pred_test)
recall = recall_score(encoded_y_test, pred_test)
f1 = f1_score(encoded_y_test, pred_test)
accuracy_train = accuracy_score(encoded_y_train, pred_train)
accuracy_test = accuracy_score(encoded_y_test, pred_test)

# Printing results
print("Accuracy train:", np.round(accuracy_train, 8))
print("Accuracy test:", np.round(accuracy_test, 8))
print("Precision:", np.round(precision, 8))
print("Recall:", np.round(recall, 8))
print("F1 Score:", np.round(f1, 8))

print('\n ---------------------------------- MULTICLASS CLASSIFICATION --------------------------------------------- \n')

nsl_kdd_train = kdd_train.copy()
nsl_kdd_test = kdd_test.copy()

X_train = nsl_kdd_train[num_features]
X_test = nsl_kdd_test[num_features]
y_train = nsl_kdd_train["outcome"]
y_test = nsl_kdd_test["outcome"]

label_encoder = LabelEncoder()

# Fit the encoder on the labels and transform the column
y_train = label_encoder.fit_transform(y_train)
print(y_train)

# Fit the encoder on the labels and transform the column
y_test = label_encoder.fit_transform(y_test)
print(y_test)

encoded_labels_series = pd.Series(y_test)

# Assuming 'features' is your DataFrame
scaler = MinMaxScaler()

# Apply Min-Max scaling to each column separately
X_train_scaled = X_train.copy()
for column in X_train.columns:
    column_data = X_train[column].values.reshape(-1, 1)
    X_train_scaled[column] = scaler.fit_transform(column_data)

# If you want to keep the scaled data in the original DataFrame
X_train_scaled.columns = X_train.columns

# Apply Min-Max scaling to each column separately
X_test_scaled = X_test.copy()
for column in X_test.columns:
    column_data = X_test[column].values.reshape(-1, 1)
    X_test_scaled[column] = scaler.fit_transform(column_data)

# If you want to keep the scaled data in the original DataFrame
X_test_scaled.columns = X_test.columns

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

from collections import Counter

frequency = Counter(pred_test)

for entry, count in frequency.items():
    print(entry, ":", count)

print('------------')
frequency = Counter(y_test)

for entry, count in frequency.items():
    print(entry, ":", count)

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

forest_importances = pd.Series(importances, index=X_train.columns)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()


# Create a DataFrame with feature names and importances
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})

# Sort the DataFrame by importance scores in descending order
feature_importance_df_sorted = feature_importance_df.sort_values(by='Importance', ascending=False)

def select_important_features(feature_importance_df_sorted, threshold=0.01):
    important_features = feature_importance_df_sorted[feature_importance_df_sorted['Importance'] > threshold]
    selected_feature_names = important_features['Feature'].tolist()
    return selected_feature_names

# Usage:
selected_features = select_important_features(feature_importance_df_sorted, threshold=0.01)
print("Selected features:", selected_features)

X_train = X_train_scaled[selected_features]
X_test = X_test_scaled[selected_features]
y_train = nsl_kdd_train["outcome"]
y_test = nsl_kdd_test["outcome"]

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

pca = PCA().fit(X_train_scaled)

index_99_percent = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95)
index_99_percent = index_99_percent + 1

pca = PCA(n_components=index_99_percent)
pca_train = pca.fit_transform(X_train_scaled)
pca_test = pca.transform(X_test_scaled)

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
print ("Tets set prediction: {}s".format(round(tt, 3)))

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
print("Number of components: ", index_99_percent)