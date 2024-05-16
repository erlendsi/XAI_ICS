import pandas as pd
from sklearn.decomposition import PCA
from time import time
import numpy as np
import matplotlib.pyplot as plt; plt.style.use('dark_background')
import seaborn as sns
from tqdm import tqdm
import warnings; warnings.filterwarnings('ignore')
import psutil


from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectPercentile, f_classif, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from functions import pca_anomaly_detector, evaluate_model, get_mse_loss, get_mae_loss, get_rmse_loss
DATASET = 'KDD99'
THRESHOLD = 0.0005
COMPONENTS = 10
TITLE = f'PCA & RF: {DATASET}'
SAVE_PATH = f'images/{DATASET}_pca_rf_confusion_matrix.png'
LOSS_FUNCTION = get_mse_loss

print('\n ----- KDD99 ----- \n')


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

scaler = MinMaxScaler()

features_scaled = features.copy()
for column in features.columns:
    column_data = features[column].values.reshape(-1, 1)
    features_scaled[column] = scaler.fit_transform(column_data)

features_scaled.columns = features.columns

y_true = kdd_df['label'].copy()
print(y_true.value_counts())

y_bin = kdd_df['label'].copy()
y_bin[y_bin != 'normal.'] = 'attack.'

y_bin = label_binarize(y_bin, classes = ['normal.', 'attack.'])

pca = PCA(n_components=COMPONENTS, random_state=0)
pca_df = pd.DataFrame(pca.fit_transform(features_scaled))
df_restored = pd.DataFrame(pca.inverse_transform(pca_df), index=features_scaled.index)

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(df_restored, y_bin, test_size=0.15, random_state=42)
X_train_org, X_test_org, y_train_org, y_test_org = train_test_split(features_scaled, y_true, test_size=0.15, random_state=42)

print('\n ----- Training RF model ----- \n')

clf = RandomForestClassifier(random_state = 0)
t0 = time()
clf.fit(X_train_org, y_train_org)
tt = time() - t0
print ("Trained in {} seconds".format(round(tt,3)))

print('\n ----- Predicting Anomalies ----- \n')

y_pred = pca_anomaly_detector(X_test_pca, X_test_org, THRESHOLD, LOSS_FUNCTION)

print('\n ----- Explaining Anomalies ----- \n')

y_rf_pred = []

for i, pred_label in enumerate(y_pred):
    if pred_label == 1:
        pred = clf.predict(X_test_org.iloc[i].array.reshape(1, -1))
        y_rf_pred.append(pred[0])
    else:
        y_rf_pred.append('normal.')

print('\n ----- Results ----- \n') 

evaluate_model(y_test_org, y_rf_pred, features.columns, TITLE, SAVE_PATH)

from collections import Counter

print("Confusion Matrix:", confusion_matrix(y_test_pca, y_pred))

frequency = Counter(y_rf_pred)

for entry, count in frequency.items():
    print(entry, ":", count)

print(y_test_org.value_counts())