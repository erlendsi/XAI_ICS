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

from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectPercentile, f_classif, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from functions import pca_anomaly_detector, evaluate_model, get_mse_loss, get_mae_loss, get_rmse_loss
DATASET = 'CICIoT'
THRESHOLD = 0.0005
COMPONENTS = 12
TITLE = f'PCA & RF: {DATASET}'
SAVE_PATH = f'images/{DATASET}_pca_rf_confusion_matrix.png'
LOSS_FUNCTION = get_mae_loss

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

features = df[X_columns]
y_true = df[y_column]

y_bin = y_true.copy()

y_bin = y_bin.apply(lambda x: 'attack' if x != 'BenignTraffic' else x)
y_bin = label_binarize(y_bin, classes = ['BenignTraffic', 'attack'])


scaler = MinMaxScaler()

features_scaled = features.copy()
for column in features.columns:
    column_data = features[column].values.reshape(-1, 1)
    features_scaled[column] = scaler.fit_transform(column_data)

features_scaled.columns = features.columns

pca = PCA(n_components=COMPONENTS, random_state=0)
pca_df = pd.DataFrame(pca.fit_transform(features_scaled))
df_restored = pd.DataFrame(pca.inverse_transform(pca_df), index=features_scaled.index)

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(df_restored, y_bin, test_size=0.15, random_state=42)
X_train_org, X_test_org, y_train_org, y_test_org = train_test_split(features_scaled, y_true, test_size=0.15, random_state=42)

X_train_pca.reset_index(drop=True, inplace=True)
X_test_pca.reset_index(drop=True, inplace=True)
X_train_org.reset_index(drop=True, inplace=True)
X_test_org.reset_index(drop=True, inplace=True)

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
        y_rf_pred.append('BenignTraffic')

print('\n ----- Results ----- \n')       

evaluate_model(y_test_org, y_rf_pred, features.columns, TITLE, SAVE_PATH)

from collections import Counter

frequency = Counter(y_rf_pred)

for entry, count in frequency.items():
    print(entry, ":", count)

print(y_test_org.value_counts())