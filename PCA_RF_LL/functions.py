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
from sklearn.metrics import ConfusionMatrixDisplay 

def get_rmse_loss(df_original, df_restored):
    loss = np.sqrt(np.mean((np.array(df_original) - np.array(df_restored)) ** 2, axis=1))
    loss = pd.Series(data=loss)
    return loss

def get_mae_loss(df_original, df_restored):
    loss = np.mean(np.abs(np.array(df_original) - np.array(df_restored)), axis=1)
    loss = pd.Series(data=loss)
    return loss

def get_mse_loss(df_original, df_restored):
    loss = np.sum((np.array(df_original) - np.array(df_restored)) ** 2, axis=1)
    loss = pd.Series(data=loss)
    return loss

def pca_anomaly_detector(pca_df, org_df, loss_threshold, loss_function):
    y_pred = []
    loss_list = []
    for i in range(len(pca_df)):
        inverse_row = pca_df.iloc[i].values.reshape(1, -1) 
        org_row = org_df.iloc[i].values.reshape(1, -1) 
        loss = loss_function(org_row, inverse_row)
        loss_list.append(loss)
        if np.array(loss) > loss_threshold:
            # High Loss -> anomaly
            y_pred.append(1)
        else:       
            # Low loss -> normal traffic
            y_pred.append(0)
    return y_pred

def evaluate_model(y_true, y_pred, feature_names, title, save_path):
    # Metrics
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1= f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    
    # Print evaluation metrics
    print("Accuracy: ", np.round(accuracy, 8))
    print("Precision:", np.round(precision, 8))
    print("Recall:", np.round(recall, 8))
    print("F1 Score:", np.round(f1, 8))
    matrix = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=matrix) 
    fig, ax = plt.subplots(figsize=(20, 20))  # Adjust the figsize as needed
    disp.plot(ax=ax)
    plt.title(title)
    plt.savefig(save_path)
    plt.show()
