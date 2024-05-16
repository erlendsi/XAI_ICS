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
from functions import pca_anomaly_detector, evaluate_model, get_mse_loss, get_mae_loss, get_rmse_loss
DATASET = 'NIDS_v1'
THRESHOLD = 0.0005
COMPONENTS = 5
TITLE = f'PCA & RF: {DATASET}'
SAVE_PATH = f'images/{DATASET}_pca_rf_confusion_matrix.png'
LOSS_FUNCTION = get_mae_loss

print('\n ---------------------------------- NIDS-v1 --------------------------------------------- \n')

df = pd.read_csv("../NIDS-v1/NF-UQ-NIDS.csv")
df.describe()

X_columns = ['L4_SRC_PORT', 'L4_DST_PORT',
       'PROTOCOL', 'L7_PROTO', 'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS',
       'TCP_FLAGS', 'FLOW_DURATION_MILLISECONDS']

y_true = df["Attack"]
features = df[X_columns]

# Assuming 'features' is your DataFrame
scaler = MinMaxScaler()

# Apply Min-Max scaling to each column separately
features_scaled = features.copy()
for column in features.columns:
    column_data = features[column].values.reshape(-1, 1)
    features_scaled[column] = scaler.fit_transform(column_data)

features_scaled.columns = features.columns

pca = PCA(n_components=COMPONENTS, random_state=0)
pca_df = pd.DataFrame(pca.fit_transform(features_scaled))
df_restored = pd.DataFrame(pca.inverse_transform(pca_df), index=features_scaled.index)

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(df_restored, y_true, test_size=0.2, random_state=42)
X_train_org, X_test_org, y_train_org, y_test_org = train_test_split(features_scaled, y_true, test_size=0.2, random_state=42)

X_train_pca.reset_index(drop=True, inplace=True)
X_test_pca.reset_index(drop=True, inplace=True)
X_train_org.reset_index(drop=True, inplace=True)
X_test_org.reset_index(drop=True, inplace=True)

y_train_pca.reset_index(drop=True, inplace=True)
y_test_pca.reset_index(drop=True, inplace=True)
y_train_org.reset_index(drop=True, inplace=True)
y_test_org.reset_index(drop=True, inplace=True)

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
        y_rf_pred.append('Benign')

print('\n ----- Results ----- \n')       

evaluate_model(y_test_org, y_rf_pred, features.columns, TITLE, SAVE_PATH)

from collections import Counter

frequency = Counter(y_rf_pred)

for entry, count in frequency.items():
    print(entry, ":", count)

print(y_test_org.value_counts())