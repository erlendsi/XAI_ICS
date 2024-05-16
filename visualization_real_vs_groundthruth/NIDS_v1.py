import pandas as pd
from sklearn.decomposition import PCA
from time import time
import numpy as np
import matplotlib.pyplot as plt; plt.style.use('dark_background')
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

from collections import OrderedDict
THRESHOLD = 100
LOSS_TRHRESHOLD = 0.0005

print('\n ---------------------------------- NIDS-v1 --------------------------------------------- \n')

df = pd.read_csv("../NIDS-v1/NF-UQ-NIDS.csv", nrows=100)

X_columns = ['L4_SRC_PORT', 'L4_DST_PORT',
       'PROTOCOL', 'L7_PROTO', 'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS',
       'TCP_FLAGS', 'FLOW_DURATION_MILLISECONDS']

y_bin = df["Label"]
X_df = df[X_columns]

X = X_df[:THRESHOLD]
y = y_bin[:THRESHOLD]

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
    loss = np.mean(np.abs(np.array(df_original) - np.array(df_restored)), axis=1)
    loss = pd.Series(data=loss)
    return loss

pca = PCA(n_components=5, random_state=0)
pca_df = pd.DataFrame(pca.fit_transform(X_scaled))
df_restored = pd.DataFrame(pca.inverse_transform(pca_df), index=pca_df.index)

scores = get_anomaly_scores(X_scaled, df_restored)
scores_clipped = np.clip(scores, None, 1)

plt.figure(figsize=(16, 8))
plt.plot(scores_clipped, color="white", label="loss")
plt.axhline(y=LOSS_TRHRESHOLD, color='lime', linestyle='--', label="Loss Threshold")
for i, (score, y_val) in enumerate(zip(scores, y_encoded)):
    if score > LOSS_TRHRESHOLD and y_val == 1:
        plt.bar(i, 1, color='cyan', alpha=0.5, label="Positive")
    elif score > LOSS_TRHRESHOLD:
        plt.bar(i, 1, color='orange', alpha=0.5, label="False Positive")
    elif y_val == 1:
        plt.bar(i, 1, color='red', alpha=0.5, label="False Negative")


plt.title("NIDS-V1 - Prediction vs. Groundtruth, n components: {}".format(4))
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1, 1), loc="upper left")
plt.savefig('images/NIDSv1_pca_vs_groundtruth.png')
plt.show()