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

def get_mse_loss(df_original, df_restored):
    loss = np.sum((np.array(df_original) - np.array(df_restored)) ** 2, axis=1)
    loss = pd.Series(data=loss)
    return loss

def pca_anomaly_detector_mse(pca_df, org_df, loss_threshold):
    y_pred = []
    loss_list = []
    for i in range(len(pca_df)):
        inverse_row = pca_df.iloc[i].values.reshape(1, -1) 
        org_row = org_df.iloc[i].values.reshape(1, -1) 
        loss = get_mse_loss(org_row, inverse_row)
        loss_list.append(loss)
        if np.array(loss) > loss_threshold:
            # High Loss -> anomaly
            y_pred.append(1)
        else:       
            # Low loss -> normal traffic
            y_pred.append(0)
    return y_pred, loss_list

def get_mae_loss(df_original, df_restored):
    loss = np.mean(np.abs(np.array(df_original) - np.array(df_restored)), axis=1)
    loss = pd.Series(data=loss)
    return loss

def pca_anomaly_detector_mae(pca_df, org_df, loss_threshold):
    y_pred = []
    loss_list = []
    for i in range(len(pca_df)):
        inverse_row = pca_df.iloc[i].values.reshape(1, -1) 
        org_row = org_df.iloc[i].values.reshape(1, -1) 
        loss = get_mae_loss(org_row, inverse_row)
        loss_list.append(loss)
        if np.array(loss) > loss_threshold:
            # High Loss -> anomaly
            y_pred.append(1)
        else:       
            # Low loss -> normal traffic
            y_pred.append(0)
    return y_pred, loss_list

def get_rmse_loss(df_original, df_restored):
    loss = np.sqrt(np.mean((np.array(df_original) - np.array(df_restored)) ** 2, axis=1))
    loss = pd.Series(data=loss)
    return loss

def pca_anomaly_detector_rmse(pca_df, org_df, loss_threshold):
    y_pred = []
    loss_list = []
    for i in range(len(pca_df)):
        inverse_row = pca_df.iloc[i].values.reshape(1, -1) 
        org_row = org_df.iloc[i].values.reshape(1, -1) 
        loss = get_rmse_loss(org_row, inverse_row)
        loss_list.append(loss)
        if np.array(loss) > loss_threshold:
            # High Loss -> anomaly
            y_pred.append(1)
        else:       
            # Low loss -> normal traffic
            y_pred.append(0)
    return y_pred, loss_list

def evaluate_model(y_true, y_pred, loss_threshold, n_components):
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    accuracy = accuracy_score(y_true, y_pred)

    # Print evaluation metrics
    print("Accuracy: ", np.round(accuracy, 8))
    print("Precision:", np.round(precision, 8))
    print("Recall:", np.round(recall, 8))
    print("F1 Score:", np.round(f1, 8))
    print("Confusion Matrix:", confusion_matrix(y_true, y_pred))
    print('Threshold', loss_threshold)
    print('N components', n_components)

def plot_results(loss_or_pred, y_true, label, title, save_path):
    plt.figure(figsize=(15, 7))
    plt.plot(loss_or_pred, label=label)
    plt.plot(y_true, alpha=0.5, color='red', label='Groundtruth')
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.show()

def plot_results_2(loss_or_pred, y_true, label, title, save_path):
    plt.figure(figsize=(15, 7))
    plt.plot(loss_or_pred, label=label)
    for i, val in enumerate(y_true):
        if val == 5:
            plt.bar(i, val, color='red', alpha=0.5)
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.show()