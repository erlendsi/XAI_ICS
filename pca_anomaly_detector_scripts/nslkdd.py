import pandas as pd
from sklearn.decomposition import PCA
from time import time
import numpy as np
import matplotlib.pyplot as plt; plt.style.use('dark_background')
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectPercentile, f_classif, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from functions import get_mse_loss, pca_anomaly_detector_mse, get_mae_loss, pca_anomaly_detector_mae, get_rmse_loss, pca_anomaly_detector_rmse, evaluate_model, plot_results_2
FILE_NAME = 'NSLKDD'

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

#kdd_train = pd.read_csv("../NSL_KDD/KDDTrain+.txt", names = col_names)
kdd_test = pd.read_csv("../NSL_KDD/KDDTest+.txt", names = col_names)

#nsl_kdd_train = kdd_train.copy()
nsl_kdd_test = kdd_test.copy()

# nsl_kdd_train.loc[nsl_kdd_train['outcome'] == "normal", "outcome"] = 'normal'
# nsl_kdd_train.loc[nsl_kdd_train['outcome'] != 'normal', "outcome"] = 'attack'

nsl_kdd_test.loc[nsl_kdd_test['outcome'] == "normal", "outcome"] = 'normal'
nsl_kdd_test.loc[nsl_kdd_test['outcome'] != 'normal', "outcome"] = 'attack'

#X_train = nsl_kdd_train[num_features]
features = nsl_kdd_test[num_features]
#y_train = nsl_kdd_train["outcome"]
labels_binary = nsl_kdd_test["outcome"]

#X = pd.concat([X_train, X_test])

#y = pd.concat([y_train, y_test])

#X.reset_index(drop=True, inplace=True)
#y.reset_index(drop=True, inplace=True)
#print(y.value_counts())

scaler = MinMaxScaler()

features_scaled = features.copy()
for column in features.columns:
    column_data = features[column].values.reshape(-1, 1)
    features_scaled[column] = scaler.fit_transform(column_data)

features_scaled.columns = features.columns

y_true = label_binarize(labels_binary, classes = ['normal', 'attack'])
unique_labels, counts = np.unique(y_true, return_counts=True)

for label, count in zip(unique_labels, counts):
    print(f"Label: {label}, Count: {count}")

print('\n ---------------------------------- Finding 95% & 99% --------------------------------------------- \n')


pca = PCA(whiten=True).fit(features_scaled)

# Find the index where cumulative explained variance first exceeds 95%
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

print('\n -- MAE -- \n')

thresholds = [0.05, 0.005, 0.0005]
for threshold in thresholds:
    y_pred_mae, loss_mae = pca_anomaly_detector_mae(df_restored, features_scaled, threshold)
    evaluate_model(y_true, y_pred_mae, threshold, index_95_percent)

print('\n -- RMSE -- \n')

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