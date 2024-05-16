import pandas as pd
from sklearn.decomposition import PCA
from time import time
import numpy as np
import matplotlib.pyplot as plt; plt.style.use('dark_background')
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
POSITIVE = 5
NEGATIVE = 1

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
kdd_df = kdd_df.iloc[:100]

features = kdd_df[num_features].astype(float)

scaler = MinMaxScaler()

features_scaled = features.copy()
for column in features.columns:
    column_data = features[column].values.reshape(-1, 1)
    features_scaled[column] = scaler.fit_transform(column_data)

features_scaled.columns = features.columns

y_true = kdd_df['label'].copy()

label_encoder = LabelEncoder()

encoded_labels = y_true.copy()

# Fit the encoder on the labels and transform the column
encoded_labels = label_encoder.fit_transform(encoded_labels)

print(features_scaled.shape)
print(encoded_labels.shape)

X_train, X_test, y_train, y_test = train_test_split(features_scaled, encoded_labels, test_size=0.15, random_state=42)

#Training a classifier
clf = RandomForestClassifier(random_state=0)
t0 = time()
clf.fit(X_train, y_train)
tt = time() - t0
print ("Classifier trained in {} seconds.".format(round(tt, 3)))

exm = ExplainableMatrix(n_features=len(num_features), n_classes=len(np.unique(encoded_labels)) , feature_names=num_features, class_names=y_true.unique())
exm.rules_extration(clf, X_test.to_numpy(), y_test, clf.feature_importances_)
print( 'n_rules DT', exm.n_rules_ )

# Prediction on test set
t0 = time()
pred_test = clf.predict(X_test)
tt = time() - t0
print ("Classifier predicted on test set in {} seconds.".format(round(tt, 3)))

# Metrics
precision = precision_score(y_test, pred_test, average='weighted')
recall = recall_score(y_test, pred_test, average='weighted')
f1 = f1_score(y_test, pred_test, average='weighted')
accuracy_test = accuracy_score(y_test, pred_test)

# Printing results
print("Accuracy test: ", accuracy_test)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)


print(X_test.to_numpy().shape)
print(y_test.shape)


print('\n -------------- ExMatrix ---------------\n')


exm = ExplainableMatrix(n_features=len(num_features), n_classes=len(np.unique(encoded_labels)) , feature_names=num_features, class_names=y_true.unique())
exm.rules_extration(clf, X_test.to_numpy(), y_test, clf.feature_importances_)
print( 'n_rules DT', exm.n_rules_ )

#exp = exm.explanation( exp_type = 'local-used', x_k = X_test[ 13 ], r_order = 'support', f_order = 'importance', info_text = '\ninstance 13\n' )

exp = exm.explanation( exp_type = 'local-used', x_k = X_test.iloc[NEGATIVE], r_order = 'support', f_order = 'importance', info_text = '\ninstance 2\n' )
exp.create_svg( draw_x_k = False, draw_row_labels = False, draw_col_labels = True, draw_rows_line = False, draw_cols_line = True, col_label_degrees = 30, width = 2200, height = 1200, margin_bottom = 300 )
exp.save( 'NEG_ORG_1.svg' )
exp.display_jn()


exp = exm.explanation( exp_type = 'local-closest', x_k = X_test.iloc[NEGATIVE], r_order = 'delta change', f_order = 'importance', info_text = '\ninstance 2\n' )
#exp.create_svg( draw_x_k = False, draw_row_labels = False, draw_col_labels = True, draw_rows_line = False, draw_cols_line = True, col_label_degrees = 30, width = 1890, height = 600, margin_bottom = 150 )
exp.create_svg( draw_x_k = True, draw_deltas = True, cell_background = 'used', draw_row_labels = False, draw_col_labels = True, draw_rows_line = False, draw_cols_line = True, col_label_degrees = 30, width = 2200, height = 1200, margin_bottom = 300 )
exp.save( 'NEG_ORG_2.svg' )
exp.display_jn()


print('\n -------------- SHAP ---------------\n')
explainer = shap.TreeExplainer(clf)
choosen_instance = X_test.iloc[[NEGATIVE]]
shap_values = explainer.shap_values(choosen_instance)


shap.force_plot(explainer.expected_value[1], shap_values[0][:, 1], choosen_instance.iloc[0], matplotlib=True)
plt.savefig("NEG_ORG_3.svg" )
plt.show()

print('\n -------------- FEATURE SELECTION ---------------\n')

importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)

forest_importances = pd.Series(importances, index=features.columns)

# Create a DataFrame with feature names and importances
feature_importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': importances})

# Sort the DataFrame by importance scores in descending order
feature_importance_df_sorted = feature_importance_df.sort_values(by='Importance', ascending=False)

def select_important_features(feature_importance_df_sorted, threshold=0.01):
    important_features = feature_importance_df_sorted[feature_importance_df_sorted['Importance'] > threshold]
    selected_feature_names = important_features['Feature'].tolist()
    return selected_feature_names

# Usage:
selected_features = select_important_features(feature_importance_df_sorted, threshold=0.1)
print("Selected features:", selected_features)

X_train, X_test, y_train, y_test = train_test_split(features_scaled[selected_features], encoded_labels, test_size=0.15, random_state=42)

#Training a classifier
clf = RandomForestClassifier(random_state=0)
t0 = time()
clf.fit(X_train, y_train)
tt = time() - t0
print ("Classifier trained in {} seconds.".format(round(tt, 3)))

# Prediction on test set
t0 = time()
pred_test = clf.predict(X_test)
tt = time() - t0
print ("Classifier predicted on test set in {} seconds.".format(round(tt, 3)))

# Metrics
precision = precision_score(y_test, pred_test, average='weighted')
recall = recall_score(y_test, pred_test, average='weighted')
f1 = f1_score(y_test, pred_test, average='weighted')
accuracy_test = accuracy_score(y_test, pred_test)

# Printing results
print("Accuracy test: ", accuracy_test)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)

print('\n -------------- ExMatrix ---------------\n')


exm = ExplainableMatrix(n_features=len(num_features), n_classes=len(np.unique(encoded_labels)) , feature_names=selected_features, class_names=y_true.unique())
exm.rules_extration(clf, X_test.to_numpy(), y_test, clf.feature_importances_)
print( 'n_rules DT', exm.n_rules_ )

#exp = exm.explanation( exp_type = 'local-used', x_k = X_test[ 13 ], r_order = 'support', f_order = 'importance', info_text = '\ninstance 13\n' )

exp = exm.explanation( exp_type = 'local-used', x_k = X_test.iloc[NEGATIVE], r_order = 'support', f_order = 'importance', info_text = '\ninstance 2\n' )
exp.create_svg( draw_x_k = False, draw_row_labels = False, draw_col_labels = True, draw_rows_line = False, draw_cols_line = True, col_label_degrees = 30, width = 2200, height = 1200, margin_bottom = 300 )
exp.save( 'NEG_FS_1.svg' )
exp.save( 'NEG_FS_1.png' )
exp.display_jn()


exp = exm.explanation( exp_type = 'local-closest', x_k = X_test.iloc[NEGATIVE], r_order = 'delta change', f_order = 'importance', info_text = '\ninstance 2\n' )
#exp.create_svg( draw_x_k = False, draw_row_labels = False, draw_col_labels = True, draw_rows_line = False, draw_cols_line = True, col_label_degrees = 30, width = 1890, height = 600, margin_bottom = 150 )
exp.create_svg( draw_x_k = True, draw_deltas = True, cell_background = 'used', draw_row_labels = True, draw_col_labels = True, draw_rows_line = False, draw_cols_line = True, col_label_degrees = 30, width = 2200, height = 1200, margin_bottom = 300 )
exp.save( 'NEG_FS_2.svg' )
exp.save( 'NEG_FS_2.png' )
exp.display_jn()


print('\n -------------- SHAP ---------------\n')
explainer = shap.TreeExplainer(clf)
choosen_instance = X_test.iloc[[NEGATIVE]]
shap_values = explainer.shap_values(choosen_instance)

shap.force_plot(explainer.expected_value[1], shap_values[0][:, 1], choosen_instance.iloc[0], matplotlib=True)
plt.savefig("NEG_FS_3.svg" )
plt.savefig("NEG_FS_3.png" )
plt.show()

print('\n -------------- PCA ---------------\n')
pca = PCA().fit(features_scaled)
index_99_percent = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95)
index_99_percent = index_99_percent + 1

pca = PCA(n_components=index_99_percent)
pca_result = pca.fit_transform(features_scaled)

X_train, X_test, y_train, y_test = train_test_split(pca_result, encoded_labels, test_size=0.15, random_state=42)

#Training a classifier
clf = RandomForestClassifier(random_state = 69)
t0 = time()
clf.fit(X_train, y_train)
tt = time() - t0

# Prediction on train set
t0 = time()
pred_train = clf.predict(X_train)
tt = time() - t0

# Prediction on test set
t0 = time()
pred_test = clf.predict(X_test)
tt = time() - t0
print ("Train set prediction: {}s".format(round(tt, 3)))

# Metrics
precision = precision_score(y_test, pred_test, average='weighted')
recall = recall_score(y_test, pred_test, average='weighted')
f1 = f1_score(y_test, pred_test, average='weighted')
accuracy_train = accuracy_score(y_train, pred_train)
accuracy_test = accuracy_score(y_test, pred_test)

# Printing results
print("Accuracy train:", np.round(accuracy_train, 8))
print("Accuracy test:", np.round(accuracy_test, 8))
print("Precision:", np.round(precision, 8))
print("Recall:", np.round(recall, 8))
print("F1 Score:", np.round(f1, 8))
print("Number of components:", index_99_percent)

print('\n -------------- ExMatrix ---------------\n')

exm = ExplainableMatrix(n_features=len(num_features), n_classes=len(np.unique(encoded_labels)), class_names=y_true.unique())
exm.rules_extration(clf, X_test, y_test, clf.feature_importances_)
print( 'n_rules DT', exm.n_rules_ )

#exp = exm.explanation( exp_type = 'local-used', x_k = X_test[ 13 ], r_order = 'support', f_order = 'importance', info_text = '\ninstance 13\n' )

exp = exm.explanation( exp_type = 'local-used', x_k = X_test[NEGATIVE], r_order = 'support', f_order = 'importance', info_text = '\ninstance 2\n' )
exp.create_svg( draw_x_k = False, draw_row_labels = False, draw_col_labels = False, draw_rows_line = False, draw_cols_line = True, col_label_degrees = 30, width = 2200, height = 1200, margin_bottom = 300 )
exp.save( 'NEG_PCA_1.svg' )
exp.display_jn()


exp = exm.explanation( exp_type = 'local-closest', x_k = X_test[NEGATIVE], r_order = 'delta change', f_order = 'importance', info_text = '\ninstance 2\n' )
#exp.create_svg( draw_x_k = False, draw_row_labels = False, draw_col_labels = True, draw_rows_line = False, draw_cols_line = True, col_label_degrees = 30, width = 1890, height = 600, margin_bottom = 150 )
exp.create_svg( draw_x_k = True, draw_deltas = True, cell_background = 'used', draw_row_labels = False, draw_col_labels = False, draw_rows_line = False, draw_cols_line = True, col_label_degrees = 30, width = 2200, height = 1200, margin_bottom = 300 )
exp.save( 'NEG_PCA_2.svg' )
exp.display_jn()


print('\n -------------- SHAP ---------------\n')
explainer = shap.TreeExplainer(clf)
choosen_instance = X_test[NEGATIVE]
shap_values = explainer.shap_values(choosen_instance)

print("Shape of choosen_instance:", choosen_instance.shape)
print("Shape of shap_values:", shap_values.shape)

shap.force_plot(explainer.expected_value[1], shap_values[:, 1], choosen_instance, matplotlib=True)
plt.savefig("NEG_PCA_3.svg" )
plt.show()

print('----------------------------------------------------------------------- POSITIVE CASE -----------------------------------------------------------------')


X_train, X_test, y_train, y_test = train_test_split(features_scaled, encoded_labels, test_size=0.15, random_state=42)

#Training a classifier
clf = RandomForestClassifier(random_state=0)
t0 = time()
clf.fit(X_train, y_train)
tt = time() - t0
print ("Classifier trained in {} seconds.".format(round(tt, 3)))

# Prediction on test set
t0 = time()
pred_test = clf.predict(X_test)
tt = time() - t0
print ("Classifier predicted on test set in {} seconds.".format(round(tt, 3)))

# Metrics
precision = precision_score(y_test, pred_test, average='weighted')
recall = recall_score(y_test, pred_test, average='weighted')
f1 = f1_score(y_test, pred_test, average='weighted')
accuracy_test = accuracy_score(y_test, pred_test)

# Printing results
print("Accuracy test: ", accuracy_test)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)

print(encoded_labels)

print('\n -------------- ExMatrix ---------------\n')


exm = ExplainableMatrix(n_features=len(num_features), n_classes=len(np.unique(encoded_labels)) , feature_names=num_features, class_names=y_true.unique())
exm.rules_extration(clf, X_test.to_numpy(), y_test, clf.feature_importances_)
print( 'n_rules DT', exm.n_rules_ )

#exp = exm.explanation( exp_type = 'local-used', x_k = X_test[ 13 ], r_order = 'support', f_order = 'importance', info_text = '\ninstance 13\n' )

exp = exm.explanation( exp_type = 'local-used', x_k = X_test.iloc[POSITIVE], r_order = 'support', f_order = 'importance', info_text = '\ninstance 2\n' )
exp.create_svg( draw_x_k = False, draw_row_labels = False, draw_col_labels = True, draw_rows_line = False, draw_cols_line = True, col_label_degrees = 30, width = 2200, height = 1200, margin_bottom = 300 )
exp.save( 'POS_ORG_1.svg' )
exp.display_jn()


exp = exm.explanation( exp_type = 'local-closest', x_k = X_test.iloc[POSITIVE], r_order = 'delta change', f_order = 'importance', info_text = '\ninstance 2\n' )
#exp.create_svg( draw_x_k = False, draw_row_labels = False, draw_col_labels = True, draw_rows_line = False, draw_cols_line = True, col_label_degrees = 30, width = 1890, height = 600, margin_bottom = 150 )
exp.create_svg( draw_x_k = True, draw_deltas = True, cell_background = 'used', draw_row_labels = False, draw_col_labels = True, draw_rows_line = False, draw_cols_line = True, col_label_degrees = 30, width = 2200, height = 1200, margin_bottom = 300 )
exp.save( 'POS_ORG_2.svg' )
exp.display_jn()


print('\n -------------- SHAP ---------------\n')
explainer = shap.TreeExplainer(clf)
choosen_instance = X_test.iloc[[POSITIVE]]
shap_values = explainer.shap_values(choosen_instance)


shap.force_plot(explainer.expected_value[1], shap_values[0][:, 1], choosen_instance.iloc[0], matplotlib=True)
plt.savefig("POS_ORG_3.svg" )
plt.show()

print('\n -------------- FEATURE SELECTION ---------------\n')

importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)

forest_importances = pd.Series(importances, index=features.columns)

# Create a DataFrame with feature names and importances
feature_importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': importances})

# Sort the DataFrame by importance scores in descending order
feature_importance_df_sorted = feature_importance_df.sort_values(by='Importance', ascending=False)

def select_important_features(feature_importance_df_sorted, threshold=0.01):
    important_features = feature_importance_df_sorted[feature_importance_df_sorted['Importance'] > threshold]
    selected_feature_names = important_features['Feature'].tolist()
    return selected_feature_names

# Usage:
selected_features = select_important_features(feature_importance_df_sorted, threshold=0.01)
print("Selected features:", selected_features)

X_train, X_test, y_train, y_test = train_test_split(features_scaled[selected_features], encoded_labels, test_size=0.15, random_state=42)

#Training a classifier
clf = RandomForestClassifier(random_state=0)
t0 = time()
clf.fit(X_train, y_train)
tt = time() - t0
print ("Classifier trained in {} seconds.".format(round(tt, 3)))

# Prediction on test set
t0 = time()
pred_test = clf.predict(X_test)
tt = time() - t0
print ("Classifier predicted on test set in {} seconds.".format(round(tt, 3)))

# Metrics
precision = precision_score(y_test, pred_test, average='weighted')
recall = recall_score(y_test, pred_test, average='weighted')
f1 = f1_score(y_test, pred_test, average='weighted')
accuracy_test = accuracy_score(y_test, pred_test)

# Printing results
print("Accuracy test: ", accuracy_test)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)

print('\n -------------- ExMatrix ---------------\n')


exm = ExplainableMatrix(n_features=len(num_features), n_classes=len(np.unique(encoded_labels)) , feature_names=selected_features, class_names=y_true.unique())
exm.rules_extration(clf, X_test.to_numpy(), y_test, clf.feature_importances_)
print( 'n_rules DT', exm.n_rules_ )

#exp = exm.explanation( exp_type = 'local-used', x_k = X_test[ 13 ], r_order = 'support', f_order = 'importance', info_text = '\ninstance 13\n' )

exp = exm.explanation( exp_type = 'local-used', x_k = X_test.iloc[POSITIVE], r_order = 'support', f_order = 'importance', info_text = '\ninstance 2\n' )
exp.create_svg( draw_x_k = False, draw_row_labels = False, draw_col_labels = True, draw_rows_line = False, draw_cols_line = True, col_label_degrees = 30, width = 2200, height = 1200, margin_bottom = 300 )
exp.save( 'POS_FS_1.svg' )
exp.display_jn()


exp = exm.explanation( exp_type = 'local-closest', x_k = X_test.iloc[POSITIVE], r_order = 'delta change', f_order = 'importance', info_text = '\ninstance 2\n' )
#exp.create_svg( draw_x_k = False, draw_row_labels = False, draw_col_labels = True, draw_rows_line = False, draw_cols_line = True, col_label_degrees = 30, width = 1890, height = 600, margin_bottom = 150 )
exp.create_svg( draw_x_k = True, draw_deltas = True, cell_background = 'used', draw_row_labels = True, draw_col_labels = True, draw_rows_line = False, draw_cols_line = True, col_label_degrees = 30, width = 2200, height = 1200, margin_bottom = 300 )
exp.save( 'POS_FS_2.svg' )
exp.display_jn()


print('\n -------------- SHAP ---------------\n')
explainer = shap.TreeExplainer(clf)
choosen_instance = X_test.iloc[[POSITIVE]]
shap_values = explainer.shap_values(choosen_instance)

shap.force_plot(explainer.expected_value[1], shap_values[0][:, 1], choosen_instance.iloc[0], matplotlib=True)
plt.savefig("POS_FS_3.svg" )
plt.show()

print('\n -------------- PCA ---------------\n')
pca = PCA().fit(features_scaled)
index_99_percent = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95)
index_99_percent = index_99_percent + 1

pca = PCA(n_components=index_99_percent)
pca_result = pca.fit_transform(features_scaled)

X_train, X_test, y_train, y_test = train_test_split(pca_result, encoded_labels, test_size=0.15, random_state=42)

#Training a classifier
clf = RandomForestClassifier(random_state = 69)
t0 = time()
clf.fit(X_train, y_train)
tt = time() - t0

# Prediction on train set
t0 = time()
pred_train = clf.predict(X_train)
tt = time() - t0

# Prediction on test set
t0 = time()
pred_test = clf.predict(X_test)
tt = time() - t0
print ("Train set prediction: {}s".format(round(tt, 3)))

# Metrics
precision = precision_score(y_test, pred_test, average='weighted')
recall = recall_score(y_test, pred_test, average='weighted')
f1 = f1_score(y_test, pred_test, average='weighted')
accuracy_train = accuracy_score(y_train, pred_train)
accuracy_test = accuracy_score(y_test, pred_test)

# Printing results
print("Accuracy train:", np.round(accuracy_train, 8))
print("Accuracy test:", np.round(accuracy_test, 8))
print("Precision:", np.round(precision, 8))
print("Recall:", np.round(recall, 8))
print("F1 Score:", np.round(f1, 8))
print("Number of components:", index_99_percent)

print('\n -------------- ExMatrix ---------------\n')


exm = ExplainableMatrix(n_features=len(num_features), n_classes=len(np.unique(encoded_labels)), class_names=y_true.unique())
exm.rules_extration(clf, X_test, y_test, clf.feature_importances_)
print( 'n_rules DT', exm.n_rules_ )

#exp = exm.explanation( exp_type = 'local-used', x_k = X_test[ 13 ], r_order = 'support', f_order = 'importance', info_text = '\ninstance 13\n' )

exp = exm.explanation( exp_type = 'local-used', x_k = X_test[POSITIVE], r_order = 'support', f_order = 'importance', info_text = '\ninstance 2\n' )
exp.create_svg( draw_x_k = False, draw_row_labels = False, draw_col_labels = False, draw_rows_line = False, draw_cols_line = True, col_label_degrees = 30, width = 2200, height = 1200, margin_bottom = 300 )
exp.save( 'POS_PCA_1.svg' )
exp.display_jn()


exp = exm.explanation( exp_type = 'local-closest', x_k = X_test[POSITIVE], r_order = 'delta change', f_order = 'importance', info_text = '\ninstance 2\n' )
#exp.create_svg( draw_x_k = False, draw_row_labels = False, draw_col_labels = True, draw_rows_line = False, draw_cols_line = True, col_label_degrees = 30, width = 1890, height = 600, margin_bottom = 150 )
exp.create_svg( draw_x_k = True, draw_deltas = True, cell_background = 'used', draw_row_labels = False, draw_col_labels = False, draw_rows_line = False, draw_cols_line = True, col_label_degrees = 30, width = 2200, height = 1200, margin_bottom = 300 )
exp.save( 'POS_PCA_2.svg' )
exp.display_jn()


print('\n -------------- SHAP ---------------\n')
explainer = shap.TreeExplainer(clf)
choosen_instance = X_test[POSITIVE]
shap_values = explainer.shap_values(choosen_instance)

shap.force_plot(explainer.expected_value[1], shap_values[:, 1], matplotlib=True)

plt.savefig("POS_PCA_3.svg" )
plt.show()
