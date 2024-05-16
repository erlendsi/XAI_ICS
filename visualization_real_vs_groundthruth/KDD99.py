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

from collections import OrderedDict
THRESHOLD = 100
LOSS_TRHRESHOLD = 0.0005

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

kdd_df = pd.read_csv("../KDD99/corrected.csv", names = col_names, nrows=100)

features = kdd_df[num_features].astype(float)

labels = kdd_df['label'].copy()
labels_binary = kdd_df['label'].copy()
labels_binary[labels_binary != 'normal.'] = 'attack.'

X = features[:THRESHOLD]
y = labels_binary[:THRESHOLD]

scaler = MinMaxScaler()

X_scaled = X.copy()
for column in X.columns:
    column_data = X[column].values.reshape(-1, 1)
    X_scaled[column] = scaler.fit_transform(column_data)

# If you want to keep the scaled data in the original DataFrame
X_scaled.columns = X.columns

label_encoder = LabelEncoder()

y_encoded = y.copy()

# Fit the encoder on the labels and transform the column
y_encoded = label_encoder.fit_transform(y_encoded)
y_encoded = y_encoded[:THRESHOLD]

def get_anomaly_scores(df_original, df_restored):
    loss = np.sum((np.array(df_original) - np.array(df_restored)) ** 2, axis=1)
    loss = pd.Series(data=loss)
    return loss

pca = PCA(n_components=10, random_state=0)
pca_df = pd.DataFrame(pca.fit_transform(X_scaled))
df_restored = pd.DataFrame(pca.inverse_transform(pca_df), index=pca_df.index)

scores = get_anomaly_scores(X_scaled, df_restored)
scores_clipped = np.clip(scores, None, 1)

plt.figure(figsize=(16, 8))
plt.plot(scores_clipped, color="white", label="Loss")
plt.axhline(y=LOSS_TRHRESHOLD, color='lime', linestyle='--', label="Loss Threshold")
for i, (score, y_val) in enumerate(zip(scores, y_encoded)):
    if score > LOSS_TRHRESHOLD and y_val == 1:
        plt.bar(i, 1, color='cyan', alpha=0.5, label="Positive")
    elif score > LOSS_TRHRESHOLD:
        plt.bar(i, 1, color='orange', alpha=0.5, label="False Positive")
    elif y_val == 1:
        plt.bar(i, 1, color='red', alpha=0.5, label="False Negative")


plt.title("KDD99 - Prediction vs. Groundtruth, n components: {}".format(5))
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1, 1), loc="upper left")
plt.savefig('images/KDD99_pca_vs_groundtruth.png')
plt.show()