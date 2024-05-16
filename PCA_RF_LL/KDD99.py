import pandas as pd
from sklearn.decomposition import PCA
from time import time
import numpy as np
import matplotlib.pyplot as plt; plt.style.use('default')
import seaborn as sns
from tqdm import tqdm
import warnings; warnings.filterwarnings('ignore')
import psutil
import shap
from exmatrix import ExplainableMatrix

from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectPercentile, f_classif, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from functions import pca_anomaly_detector, evaluate_model, get_mse_loss, get_mae_loss, get_rmse_loss

from PIL import Image, ImageDraw, ImageFont

DATASET1 = 'KDD99'
DATASET2 = 'PUMP_SENSOR'
THRESHOLD1 = 0.0005
COMPONENTS1 = 10
THRESHOLD2 = 0.05
COMPONENTS2 = 13
CASE1 = 4465 # False Data Injection
CASE2 = 3910 # Network Attack
CASE_NORMAL = 4
CASE_ATTACK = 98

TITLE1 = f'PCA & RF: {DATASET1}'
SAVE_PATH1 = f'images/{DATASET1}_pca_rf_confusion_matrix.png'
TITLE2 = f'PCA & RF: {DATASET2}'
SAVE_PATH2 = f'images/{DATASET2}_pca_rf_confusion_matrix.png'
LOSS_FUNCTION1 = get_mse_loss
LOSS_FUNCTION2 = get_mae_loss
print('\n ----- KDD99 ----- \n')

#loading the data
col_names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot", 
             "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", 
             "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", 
             "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", 
             "srv_diff_host_rate", "dst_host_count","dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", 
             "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", 
             "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]

#Initially, we will use all features
num_features = ["duration", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", 
                "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
                "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", 
                "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
                "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", 
                "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                "dst_host_rerror_rate", "dst_host_srv_rerror_rate"]

kdd_df = pd.read_csv("../KDD99/corrected.csv", names = col_names)
kdd_df = kdd_df.sample(n=10000, random_state=42) 

kdd = kdd_df[num_features].astype(float)

scaler = MinMaxScaler()

kdd_scaled = kdd.copy()
for column in kdd.columns:
    column_data = kdd[column].values.reshape(-1, 1)
    kdd_scaled[column] = scaler.fit_transform(column_data)

kdd_scaled.columns = kdd.columns

y_kdd_ = kdd_df['label'].copy()

label_encoder = LabelEncoder()

y_kdd = label_encoder.fit_transform(y_kdd_)

y_kdd_features = np.unique(y_kdd_)

pca = PCA(n_components=COMPONENTS1, random_state=0)
pca_df_net = pd.DataFrame(pca.fit_transform(kdd_scaled))
df_restored_net = pd.DataFrame(pca.inverse_transform(pca_df_net), index=kdd_scaled.index)

X_train_pca_net, X_test_pca_net, y_train_pca_net, y_test_pca_net = train_test_split(df_restored_net, y_kdd, test_size=0.15, random_state=42)
X_train_org_net, X_test_org_net, y_train_org_net, y_test_org_net = train_test_split(kdd_scaled, y_kdd, test_size=0.15, random_state=42)

class_dict = {encoded_class: original_class for encoded_class, original_class in enumerate(label_encoder.classes_)}
print(class_dict)

print(len(label_encoder.classes_))
print(len(y_kdd_features))

#-------------------------------------------------------------

pump_sensor_df = pd.read_csv('../pump_sensor_data/sensor.csv')
y_pump_sensor = pump_sensor_df["machine_status"]
pump_sensor_df.drop(['sensor_50','sensor_15', 'timestamp'] , axis = 1 , inplace = True)
pump_sensor_df = pump_sensor_df.dropna()
pump_sensor_df = pump_sensor_df.sample(n=10000, random_state=42) 
pd_x_features = pump_sensor_df.columns.drop('machine_status').tolist()
pd_x = pump_sensor_df[pd_x_features]

scaler = MinMaxScaler()

pump_scaled = pd_x.copy()
for column in pd_x.columns:
    column_data = pd_x[column].values.reshape(-1, 1)
    pump_scaled[column] = scaler.fit_transform(column_data)

pump_scaled.columns = pd_x.columns

label_encoder = LabelEncoder()

y_pump_sensor = pump_sensor_df["machine_status"]

y_pump = y_pump_sensor.copy()
y_pump = label_encoder.fit_transform(y_pump)

class_dict = {encoded_class: original_class for encoded_class, original_class in enumerate(label_encoder.classes_)}
print(class_dict)

y_pump_features = np.unique(y_pump_sensor)

pca = PCA(n_components=COMPONENTS2, random_state=0)
pca_df_phy = pd.DataFrame(pca.fit_transform(pump_scaled))
df_restored_phy = pd.DataFrame(pca.inverse_transform(pca_df_phy), index=pump_scaled.index)


X_train_pca_phy, X_test_pca_phy, y_train_pca_phy, y_test_pca_phy = train_test_split(df_restored_phy, y_pump, test_size=0.15, random_state=42)
X_train_org_phy, X_test_org_phy, y_train_org_phy, y_test_org_phy = train_test_split(pump_scaled, y_pump, test_size=0.15, random_state=42)

print('\n ----- Training RF model ----- \n')

clf_net = RandomForestClassifier(random_state = 0)
t0 = time()
clf_net.fit(X_train_org_net, y_train_org_net)
tt = time() - t0
print ("Trained in {} seconds".format(round(tt,3)))

clf_phy = RandomForestClassifier(random_state = 0)
t0 = time()
clf_phy.fit(X_train_org_phy, y_train_org_phy)
tt = time() - t0
print ("Trained in {} seconds".format(round(tt,3)))

print('\n ----- Predicting Anomalies ----- \n')

y_net_pred = pca_anomaly_detector(X_test_pca_net, X_test_org_net, THRESHOLD1, LOSS_FUNCTION1)
y_phy_pred = pca_anomaly_detector(X_test_pca_phy, X_test_org_phy, THRESHOLD2, LOSS_FUNCTION2)

print('\n ----- Explaining Anomalies ----- \n')

y_rf_pred_net = []
y_rf_pred_phy = []

for index, value in enumerate(y_net_pred):
    if value == 1:
        pred = clf_net.predict(X_test_org_net.iloc[index].array.reshape(1, -1))
        y_rf_pred_net.append(pred[0])
    elif value == 0:
        y_rf_pred_net.append(9) # 9 is the negative class after label_encode
        
for index, value in enumerate(y_phy_pred):
    if value == 1:
        pred = clf_phy.predict(X_test_org_phy.iloc[index].array.reshape(1, -1))
        y_rf_pred_phy.append(pred[0])
    elif value == 0:
        y_rf_pred_phy.append(0)

print('\n ----- Logic Layer ----- \n')
print(len(y_rf_pred_phy))
print(len(y_rf_pred_net))
logic_layer = []
for i in range(len(y_rf_pred_net)):
    network_data = y_rf_pred_net[i]
    physical_data = y_rf_pred_phy[i]
    if network_data == 1 and physical_data == 0:
        logic_layer.append('normal')
    elif network_data == 1 and physical_data != 0:
        logic_layer.append('Failing Sensor')
    elif network_data != 1 and physical_data == 0:
        logic_layer.append('Network Attack')
    elif network_data != 1 and physical_data != 0:
        logic_layer.append('False Data Injection')

print('\n ----- Results ----- \n')       

print('NET')
evaluate_model(y_test_org_net[:len(y_rf_pred_net)], y_rf_pred_net, kdd_scaled.columns, TITLE1, SAVE_PATH1)
print(f'\n PHY')
evaluate_model(y_test_org_phy[:len(y_rf_pred_phy)], y_rf_pred_phy, kdd_scaled.columns, TITLE2, SAVE_PATH2)

print(f'\n ----- Explainability ${CASE1} ----- \n')  

print('\n -------------- ExMatrix ---------------\n')

print(f'Logic Layer at data point {CASE_NORMAL} : {logic_layer[CASE_NORMAL]}')

exm = ExplainableMatrix(n_features=len(num_features), n_classes=len(y_kdd_features), feature_names=num_features, class_names=y_kdd_features)
exm.rules_extration(clf_net, X_test_org_net.to_numpy(), y_test_org_net, clf_net.feature_importances_)
print( 'n_rules DT', exm.n_rules_ )


exp = exm.explanation( exp_type = 'local-used', x_k = X_test_org_net.iloc[CASE_NORMAL], r_order = 'class & support', f_order = 'importance', info_text = f'\ninstance ${CASE_NORMAL}\n', r_support_min = 0.4)
exp.create_svg( draw_x_k = False, draw_col_labels = True, draw_cols_line = True, col_label_font_size = 18, col_label_degrees = 35, height = 1600, margin_bottom = 300, stroke_width = 1, font_size = 18 )
exp.save( 'exm_case1_net.png' )
exp.display_jn()


exm = ExplainableMatrix(n_features=len(pd_x_features), n_classes=len(y_pump_features), feature_names=pd_x_features, class_names=y_pump_features)
exm.rules_extration(clf_phy, X_test_org_phy.to_numpy(), y_test_org_phy, clf_phy.feature_importances_)
print( 'n_rules DT', exm.n_rules_ )

exp = exm.explanation( exp_type = 'local-used', x_k = X_test_org_phy.iloc[CASE_NORMAL], r_order = 'class & support', f_order = 'importance', info_text = f'\ninstance ${CASE_NORMAL}\n', r_support_min = 0.4 )
exp.create_svg( draw_x_k = False, draw_col_labels = True, draw_cols_line = True, col_label_font_size = 18, col_label_degrees = 35, height = 1600, margin_bottom = 300, stroke_width = 1, font_size = 18 )
exp.save( 'exm_case1_phy.png' )
exp.display_jn()


print('\n -------------- SHAP ---------------\n')
explainer = shap.TreeExplainer(clf_net)
choosen_instance = X_test_org_net.iloc[[CASE_NORMAL]]
shap_values = explainer.shap_values(choosen_instance)

shap.force_plot(explainer.expected_value[1], shap_values[0][:, 1], choosen_instance.iloc[0], matplotlib=True)
plt.savefig("shap_case1_net.png")
plt.show()

explainer = shap.TreeExplainer(clf_phy)
choosen_instance = X_test_org_phy.iloc[[CASE_NORMAL]]
shap_values = explainer.shap_values(choosen_instance)

shap.force_plot(explainer.expected_value[1], shap_values[0][:, 1], choosen_instance.iloc[0], matplotlib=True)
plt.savefig("shap_case1_phy.png" )
plt.show()


print(f'\n ----- Explainability ${CASE2} ----- \n')   

print('\n -------------- ExMatrix ---------------\n')

print('Logic Layer at data point 29037: ', logic_layer[CASE_ATTACK])

exm = ExplainableMatrix(n_features=len(num_features), n_classes=len(y_kdd_features), feature_names=num_features, class_names=y_kdd_features)
exm.rules_extration(clf_net, X_test_org_net.to_numpy(), y_test_org_net, clf_net.feature_importances_)
print( 'n_rules DT', exm.n_rules_ )

exp = exm.explanation( exp_type = 'local-used', x_k = X_test_org_net.iloc[CASE_ATTACK], r_order = 'class & support', f_order = 'importance', info_text = f'\ninstance ${CASE_ATTACK}\n', r_support_min = 0.3)
exp.create_svg( draw_x_k = False, draw_col_labels = True, draw_cols_line = True, col_label_font_size = 18, col_label_degrees = 35, height = 1600, margin_bottom = 300, stroke_width = 1, font_size = 18 )
exp.save( 'exm_case2_net.png' )
exp.display_jn()

exm = ExplainableMatrix(n_features=len(pd_x_features), n_classes=len(y_pump_features), feature_names=pd_x_features, class_names=y_pump_features)
exm.rules_extration(clf_phy, X_test_org_phy.to_numpy(), y_test_org_phy, clf_phy.feature_importances_)
print( 'n_rules DT', exm.n_rules_ )

exp = exm.explanation( exp_type = 'local-used', x_k = X_test_org_phy.iloc[CASE_ATTACK], r_order = 'class & support', f_order = 'importance', info_text = f'\ninstance {CASE_ATTACK}\n', r_support_min = 0.3 )
exp.create_svg( draw_x_k = False, draw_col_labels = True, draw_cols_line = True, col_label_font_size = 18, col_label_degrees = 35, height = 1600, margin_bottom = 300, stroke_width = 1, font_size = 18 )
exp.save( 'exm_case2_phy.png' )
exp.display_jn()


print('\n -------------- SHAP ---------------\n')
explainer = shap.TreeExplainer(clf_net)
choosen_instance = X_test_org_net.iloc[[CASE_ATTACK]]
shap_values = explainer.shap_values(choosen_instance)

shap.force_plot(explainer.expected_value[1], shap_values[0][:, 1], choosen_instance.iloc[0], matplotlib=True)
plt.savefig("shap_case2_net.png" )
plt.show()

explainer = shap.TreeExplainer(clf_phy)
choosen_instance = X_test_org_phy.iloc[[CASE_ATTACK]]
shap_values = explainer.shap_values(choosen_instance)

shap.force_plot(explainer.expected_value[1], shap_values[0][:, 1], choosen_instance.iloc[0], matplotlib=True)
plt.savefig("shap_case2_phy.png" )
plt.show()

print('------------------ FS ------------------')

importances = clf_net.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf_net.estimators_], axis=0)

forest_importances = pd.Series(importances, index=kdd_scaled.columns)

# Create a DataFrame with feature names and importances
feature_importance_df = pd.DataFrame({'Feature': kdd_scaled.columns, 'Importance': importances})

# Sort the DataFrame by importance scores in descending order
feature_importance_df_sorted = feature_importance_df.sort_values(by='Importance', ascending=False)

def select_important_features(feature_importance_df_sorted, threshold=0.2):
    important_features = feature_importance_df_sorted[feature_importance_df_sorted['Importance'] > threshold]
    selected_feature_names = important_features['Feature'].tolist()
    return selected_feature_names

# Usage:
selected_features = select_important_features(feature_importance_df_sorted, threshold=0.1)
print("Selected features:", selected_features)

X_train, X_test, y_train, y_test = train_test_split(kdd_scaled[selected_features], y_kdd, test_size=0.15, random_state=42)

#Training a classifier
clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)

exm = ExplainableMatrix(n_features=len(num_features), n_classes=len(y_kdd_features), feature_names=num_features, class_names=y_kdd_features)
exm.rules_extration(clf, X_test.to_numpy(), y_test, clf.feature_importances_)
print( 'n_rules DT', exm.n_rules_ )

exp = exm.explanation( exp_type = 'local-used', x_k = X_test.iloc[CASE_NORMAL], r_order = 'class & support', f_order = 'importance', info_text = f'\ninstance ${CASE_NORMAL}\n', r_support_min = 0.3)
exp.create_svg( draw_x_k = False, draw_col_labels = True, draw_cols_line = True, col_label_font_size = 18, col_label_degrees = 35, height = 1600, margin_bottom = 300, stroke_width = 1, font_size = 18 )
exp.save( 'exm_case1_FS_net.png' )
exp.display_jn()


print('\n -------------- PCA ---------------\n')
pca = PCA().fit(kdd_scaled)
index_99_percent = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95)

pca = PCA(n_components=index_99_percent)
pca_result = pca.fit_transform(kdd_scaled)

pca_features = []

for i in range(pca_result.shape[1]):
    pca_features.append(f'PCA_{i+1}')

X_train, X_test, y_train, y_test = train_test_split(pca_result, y_kdd, test_size=0.15, random_state=42)

#Training a classifier
clf = RandomForestClassifier(random_state = 42)
clf.fit(X_train, y_train)

exm = ExplainableMatrix(n_features=len(pca_result), n_classes=len(y_kdd_features), feature_names=pca_features, class_names=y_kdd_features)
exm.rules_extration(clf, X_test, y_test, clf.feature_importances_)
print( 'n_rules DT', exm.n_rules_ )

exp = exm.explanation( exp_type = 'local-used', x_k = X_test[CASE_NORMAL], r_order = 'class & support', f_order = 'importance', info_text = f'\ninstance ${CASE_NORMAL}\n', r_support_min = 0.1)
exp.create_svg( draw_x_k = False, draw_col_labels = True, draw_cols_line = True, col_label_font_size = 18, col_label_degrees = 35, height = 1600, margin_bottom = 300, stroke_width = 1, font_size = 18 )
exp.save( 'exm_case1_PCA_net.png' )
exp.display_jn()