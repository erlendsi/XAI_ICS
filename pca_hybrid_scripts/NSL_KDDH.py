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
from functions import pca_anomaly_detector, evaluate_model, get_mse_loss, get_mae_loss, get_rmse_loss
DATASET = 'NSL_KDD'
THRESHOLD = 0.005
COMPONENTS = 19
TITLE = f'PCA & RF: {DATASET}'
SAVE_PATH = f'images/{DATASET}_pca_rf_confusion_matrix.png'
LOSS_FUNCTION = get_mse_loss

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

kdd_train = pd.read_csv("../NSL_KDD/KDDTrain+.txt", names = col_names)
kdd_test = pd.read_csv("../NSL_KDD/KDDTest+.txt", names = col_names)

nsl_kdd_train = kdd_train.copy()
nsl_kdd_test = kdd_test.copy()

y_train_org = nsl_kdd_train["outcome"]
y_test_org = nsl_kdd_test["outcome"]
print(y_train_org.value_counts())
print(y_test_org.value_counts())

X_train = nsl_kdd_train[num_features]
X_test = nsl_kdd_test[num_features]
y_train = nsl_kdd_train["outcome"]
y_test = nsl_kdd_test["outcome"]

y_train = y_train.apply(lambda x: 'attack' if x != 'normal' else x)
y_test = y_test.apply(lambda x: 'attack' if x != 'normal' else x)
y_train_pca = label_binarize(y_train, classes = ['normal', 'attack'])
y_test_pca = label_binarize(y_test, classes = ['normal', 'attack'])

scaler = MinMaxScaler()

X_train_org = X_train.copy()
for column in X_train.columns:
    column_data = X_train[column].values.reshape(-1, 1)
    X_train_org[column] = scaler.fit_transform(column_data)

X_train_org.columns = X_train.columns

X_test_org = X_test.copy()
for column in X_test.columns:
    column_data = X_test[column].values.reshape(-1, 1)
    X_test_org[column] = scaler.fit_transform(column_data)

X_test_org.columns = X_test.columns

pca = PCA(n_components=COMPONENTS, random_state=0)
pca_train = pd.DataFrame(pca.fit_transform(X_train_org))
X_train_pca = pd.DataFrame(pca.inverse_transform(pca_train), index=X_train_org.index)

pca = PCA(n_components=COMPONENTS, random_state=0)
pca_test = pd.DataFrame(pca.fit_transform(X_test_org))
X_test_pca = pd.DataFrame(pca.inverse_transform(pca_test), index=X_test_org.index)

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
        y_rf_pred.append('normal')

print('\n ----- Results ----- \n')       

evaluate_model(y_test_org, y_rf_pred, X_train.columns, TITLE, SAVE_PATH)

from collections import Counter

frequency = Counter(y_rf_pred)

for entry, count in frequency.items():
    print(entry, ":", count)

print(y_test_org.value_counts())