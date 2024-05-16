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
from functions import get_mse_loss, pca_anomaly_detector_mse, get_mae_loss, pca_anomaly_detector_mae, get_rmse_loss, pca_anomaly_detector_rmse, evaluate_model, plot_results_2
FILE_NAME = 'CICIoT'

print('\n ---------------------------------- CICIoT2023 --------------------------------------------- \n')

DATASET_DIRECTORY = '../CICIoT2023/'

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

_, features, _, labels_binary = train_test_split(X_df, y_df, test_size=0.15, random_state=42)


print(labels_binary.value_counts())

labels_binary = labels_binary.apply(lambda x: 'attack' if x != 'BenignTraffic' else x)
print(labels_binary.value_counts())

scaler = MinMaxScaler()

features_scaled = features.copy()
for column in features.columns:
    column_data = features[column].values.reshape(-1, 1)
    features_scaled[column] = scaler.fit_transform(column_data)

y_true = label_binarize(labels_binary, classes = ['BenignTraffic', 'attack'])
unique_labels, counts = np.unique(y_true, return_counts=True)

for label, count in zip(unique_labels, counts):
    print(f"Label: {label}, Count: {count}")


print('\n ---------------------------------- Finding 95% & 99% --------------------------------------------- \n')


pca = PCA(whiten=True).fit(features_scaled)

index_95_percent = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95)
index_99_percent = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.99)

index_95_percent = index_95_percent +1
index_99_percent = index_99_percent +1

print('\n ----- 95% ------ \n')


print('\n -- MSE -- \n')

pca = PCA(n_components=index_95_percent, random_state=0)
pca_df = pd.DataFrame(pca.fit_transform(features_scaled))
df_restored = pd.DataFrame(pca.inverse_transform(pca_df), index=features_scaled.index)

thresholds = [0.05, 0.005, 0.0005]

for threshold in thresholds:
    y_pred_mse, loss_mse = pca_anomaly_detector_mse(df_restored, features_scaled, threshold)
    evaluate_model(y_true, y_pred_mse, threshold, index_95_percent)

del y_pred_mse, loss_mse

print('\n -- MAE -- \n')

thresholds = [0.05, 0.005, 0.0005]
for threshold in thresholds:
    y_pred_mae, loss_mae = pca_anomaly_detector_mae(df_restored, features_scaled, threshold)
    evaluate_model(y_true, y_pred_mae, threshold, index_95_percent)

del y_pred_mae, loss_mae

print('\n -- RMSE -- \n')

thresholds = [0.05, 0.005, 0.0005]
for threshold in thresholds:
    y_pred_rmse, loss_rmse = pca_anomaly_detector_rmse(df_restored, features_scaled, threshold)
    evaluate_model(y_true, y_pred_rmse, threshold, index_95_percent)

del y_pred_rmse, loss_rmse, pca, pca_df, df_restored

print('\n ----- 99% ------ \n')


print('\n -- MSE -- \n')

pca = PCA(n_components=index_99_percent, random_state=0)
pca_df = pd.DataFrame(pca.fit_transform(features_scaled))
df_restored = pd.DataFrame(pca.inverse_transform(pca_df), index=features_scaled.index)

thresholds = [0.05, 0.005, 0.0005]
for threshold in thresholds:
    y_pred_mse, loss_mse = pca_anomaly_detector_mse(df_restored, features_scaled, threshold)
    evaluate_model(y_true, y_pred_mse, threshold, index_99_percent)

del y_pred_mse, loss_mse

print('\n -- MAE -- \n')

thresholds = [0.05, 0.005, 0.0005]
for threshold in thresholds:
    y_pred_mae, loss_mae = pca_anomaly_detector_mae(df_restored, features_scaled, threshold)
    evaluate_model(y_true, y_pred_mae, threshold, index_99_percent)

del y_pred_mae, loss_mae

print('\n -- RMSE -- \n')

thresholds = [0.05, 0.005, 0.0005]
for threshold in thresholds:
    y_pred_rmse, loss_rmse = pca_anomaly_detector_rmse(df_restored, features_scaled, threshold)
    evaluate_model(y_true, y_pred_rmse, threshold, index_99_percent)

del y_pred_rmse, loss_rmse, pca, pca_df, df_restored

print('\n -- 50% of all Features -- \n')

pca = PCA(n_components=(len(features_scaled.columns)//2), random_state=0)
pca_df = pd.DataFrame(pca.fit_transform(features_scaled))
df_restored = pd.DataFrame(pca.inverse_transform(pca_df), index=features_scaled.index)

print('\n -- MSE -- \n')

thresholds = [0.05, 0.005, 0.0005]
for threshold in thresholds:
    y_pred_mse, loss_mse = pca_anomaly_detector_mse(df_restored, features_scaled, threshold)
    evaluate_model(y_true, y_pred_mse, threshold, (len(features_scaled.columns)//2))

del y_pred_mse, loss_mse

print('\n -- MAE -- \n')

thresholds = [0.05, 0.005, 0.0005]
for threshold in thresholds:
    y_pred_mae, loss_mae = pca_anomaly_detector_mae(df_restored, features_scaled, threshold)
    evaluate_model(y_true, y_pred_mae, threshold, (len(features_scaled.columns)//2))
   
del y_pred_mae, loss_mae

print('\n -- RMSE -- \n')

thresholds = [0.05, 0.005, 0.0005]
for threshold in thresholds:
    y_pred_rmse, loss_rmse = pca_anomaly_detector_rmse(df_restored, features_scaled, threshold)
    evaluate_model(y_true, y_pred_rmse, threshold, (len(features_scaled.columns)//2))