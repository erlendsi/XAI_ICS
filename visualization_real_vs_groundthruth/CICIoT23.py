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

print('\n ---------------------------------- CICIoT2023 --------------------------------------------- \n')

DATASET_DIRECTORY = '../CICIoT2023/'
print(os.listdir(DATASET_DIRECTORY))

df_sets = [k for k in os.listdir(DATASET_DIRECTORY) if k.endswith('.csv')]
df_sets.sort()
df_sets = df_sets[:5]

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

y_df_binary = y_df.apply(lambda x: 'attack' if x != 'BenignTraffic' else x)
y_df_binary.value_counts()

X = X_df[:THRESHOLD]
y = y_df_binary[:THRESHOLD]

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

#  loss = np.sum((np.array(df_original) - np.array(df_restored)) ** 2, axis=1)
#loss = pd.Series(data=loss, index=df_original.index)
    #return loss
pca = PCA(n_components=12, random_state=0)
pca_df = pd.DataFrame(pca.fit_transform(X_scaled))
df_restored = pd.DataFrame(pca.inverse_transform(pca_df), index=pca_df.index)

scores = get_anomaly_scores(X_scaled, df_restored)

plt.figure(figsize=(16, 8))
plt.plot(scores, color="white", label="Loss")
plt.axhline(y=LOSS_TRHRESHOLD, color='lime', linestyle='--', label="Loss Threshold")
for i, (score, y_val) in enumerate(zip(scores, y_encoded)):
    if score > LOSS_TRHRESHOLD and y_val == 1:
        plt.bar(i, 1, color='cyan', alpha=0.5, label="Positive")
    elif score > LOSS_TRHRESHOLD:
        plt.bar(i, 1, color='orange', alpha=0.5, label="False Positive")
    elif y_val == 1:
        plt.bar(i, 1, color='red', alpha=0.5, label="False Negative")


plt.title("CICIoT - Prediction vs. Groundtruth, n components: {}".format(11))
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1, 1), loc="upper left")
plt.savefig('images/CICIoT_pca_vs_groundtruth.png')
plt.show()