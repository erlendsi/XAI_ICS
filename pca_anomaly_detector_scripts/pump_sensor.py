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
from functions import get_mse_loss, pca_anomaly_detector_mse, get_mae_loss, pca_anomaly_detector_mae, get_rmse_loss, pca_anomaly_detector_rmse, evaluate_model, plot_results
DATASET = 'PUMP_SENSOR'
THRESHOLD = 0.08
COMPONENTS = 5
TITLE = f'PCA & RF: {DATASET}'
SAVE_PATH1 = f'images/{DATASET}_pca_rf_confusion_matrix.png'
LOSS_FUNCTION = get_mse_loss

print('\n ----- PUMP SENSOR ----- \n')

pump_sensor_df = pd.read_csv('../pump_sensor_data/sensor.csv')
pump_sensor_df.drop(['sensor_50','sensor_15', 'timestamp'] , axis = 1 , inplace = True)
pump_sensor_df = pump_sensor_df.dropna()
pd_x_features = pump_sensor_df.columns.drop('machine_status').tolist()
pd_x = pump_sensor_df[pd_x_features]
pump_sensor_df[pump_sensor_df != 'NORMAL'] = 'ATTACK'
pd_y = pump_sensor_df["machine_status"]
print(pd_y.value_counts())

_, features, _, labels_binary = train_test_split(pd_x, pd_y, test_size=0.15, random_state=42)

unique_labels, counts = np.unique(labels_binary, return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"Label: {label}, Count: {count}")
    
scaler = MinMaxScaler()

features_scaled = features.copy()
for column in features.columns:
    column_data = features[column].values.reshape(-1, 1)
    features_scaled[column] = scaler.fit_transform(column_data)

features_scaled.columns = features.columns

y_true = label_binarize(labels_binary, classes = ['NORMAL', 'ATTACK'])
unique_labels, counts = np.unique(y_true, return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"Label: {label}, Count: {count}")

print('\n -- Finding 95% & 99% -- \n')

pca = PCA(whiten=True).fit(features_scaled)

index_95_percent = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95)
index_99_percent = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.99)

index_95_percent = index_95_percent + 1
index_99_percent = index_99_percent + 1

print('\n ----- 95% ------ \n')

print('\n -- MSE -- \n')

pca = PCA(n_components=index_95_percent, random_state=0)
pca_df = pd.DataFrame(pca.fit_transform(features_scaled))
df_restored = pd.DataFrame(pca.inverse_transform(pca_df), index=features_scaled.index)

thresholds = [0.05, 0.005, 0.0005]
for threshold in thresholds:
    y_pred_mse, loss_mse = pca_anomaly_detector_mse(df_restored, features_scaled, threshold)
    evaluate_model(y_true, y_pred_mse, threshold, index_95_percent)

print('\n -- MAE -- \n')

thresholds = [0.05, 0.005, 0.0005]
for threshold in thresholds:
    y_pred_mae, loss_mae = pca_anomaly_detector_mae(df_restored, features_scaled, threshold)
    evaluate_model(y_true, y_pred_mae, threshold, index_95_percent)


thresholds = [0.05, 0.005, 0.0005]
for threshold in thresholds:
    y_pred_rmse, loss_rmse = pca_anomaly_detector_rmse(df_restored, features_scaled, threshold)
    evaluate_model(y_true, y_pred_rmse, threshold, index_95_percent)

print('\n ----- 99% ------ \n')


print('\n -- MSE -- \n')

pca = PCA(n_components=index_99_percent, random_state=0)
pca_df = pd.DataFrame(pca.fit_transform(features_scaled))
df_restored = pd.DataFrame(pca.inverse_transform(pca_df), index=features_scaled.index)

thresholds = [0.05, 0.005, 0.0005]
for threshold in thresholds:
    y_pred_mse, loss_mse = pca_anomaly_detector_mse(df_restored, features_scaled, threshold)
    evaluate_model(y_true, y_pred_mse, threshold, index_99_percent)


print('\n -- MAE -- \n')

thresholds = [0.05, 0.005, 0.0005]
for threshold in thresholds:
    y_pred_mae, loss_mae = pca_anomaly_detector_mae(df_restored, features_scaled, threshold)
    evaluate_model(y_true, y_pred_mae, threshold, index_99_percent)



print('\n -- RMSE -- \n')

thresholds = [0.05, 0.005, 0.0005]
for threshold in thresholds:
    y_pred_rmse, loss_rmse = pca_anomaly_detector_rmse(df_restored, features_scaled, threshold)
    evaluate_model(y_true, y_pred_rmse, threshold, index_99_percent)



print('\n -- 50% of all Features -- \n')

pca = PCA(n_components=(len(features_scaled.columns)//2), random_state=0)
pca_df = pd.DataFrame(pca.fit_transform(features_scaled))
df_restored = pd.DataFrame(pca.inverse_transform(pca_df), index=features_scaled.index)


print('\n -- MSE -- \n')

thresholds = [0.05, 0.005, 0.0005]
for threshold in thresholds:
    y_pred_mse, loss_mse = pca_anomaly_detector_mse(df_restored, features_scaled, threshold)
    evaluate_model(y_true, y_pred_mse, threshold, (len(features_scaled.columns)//2))

print('\n -- MAE -- \n')

thresholds = [0.05, 0.005, 0.0005]
for threshold in thresholds:
    y_pred_mae, loss_mae = pca_anomaly_detector_mae(df_restored, features_scaled, threshold)
    evaluate_model(y_true, y_pred_mae, threshold, (len(features_scaled.columns)//2))

print('\n -- RMSE -- \n')

thresholds = [0.05, 0.005, 0.0005]
for threshold in thresholds:
    y_pred_rmse, loss_rmse = pca_anomaly_detector_rmse(df_restored, features_scaled, threshold)
    evaluate_model(y_true, y_pred_rmse, threshold, (len(features_scaled.columns)//2))