import pandas as pd
import numpy as numpy
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, matthews_corrcoef, precision_score, recall_score, f1_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# read data
data = pd.read_csv("creditcard.csv")
# print(data.describe())


# preprocessing
new_df = data.copy()
new_df['Amount'] = RobustScaler().fit_transform(new_df['Amount'].to_numpy().reshape(-1, 1))
new_df['Time'] = (new_df['Time'] - new_df['Time'].min()) / (new_df['Time'].max() - new_df['Time'].min())


# splitting training/testing sets
new_df = new_df.sample(frac=1, random_state=1) # shuffle rows
train, test, val = new_df[:240000], new_df[240000:262000], new_df[262000:]
train_np, test_np, val_np = train.to_numpy(), test.to_numpy(), val.to_numpy()

x_train, y_train = train_np[:,:-1], train_np[:, -1]
x_test, y_test = test_np[:,:-1], test_np[:, -1]
x_val, y_val = val_np[:,:-1], val_np[:, -1]

    # balancing data (undersampling)
fraud = new_df.query('Class == 1')
not_fraud = new_df.query('Class == 0')

balanced_df = pd.concat([fraud, not_fraud.sample(len(fraud), random_state=1)]) # making df with same amt of fraud as not fraud
balanced_df = balanced_df.sample(frac=1, random_state=1) # shuffle again
balanced_df_np = balanced_df.to_numpy()

x_train_b, y_train_b = balanced_df_np[:700, :-1], balanced_df_np[:700, -1] #rows, then columns
x_test_b, y_test_b = balanced_df_np[700:842, :-1], balanced_df_np[700:842, -1]
x_val_b, y_val_b = balanced_df_np[842:, :-1], balanced_df_np[842:, -1]


# creating model
bst = XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.3, max_delta_step=2, objective='binary:logistic') # uses full dataset
bst.fit(x_train, y_train)

bst_b = XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.3, max_delta_step=2, objective='binary:logistic') # uses balanced dataset
bst_b.fit(x_train_b, y_train_b)

# model evaluation
print("IMBALANCED REPORT - validation data\n",classification_report(y_val, bst.predict(x_val), target_names=['Not Fraud', 'Fraud']))

print("BALANCED REPORT - validation data\n",classification_report(y_val_b, bst_b.predict(x_val_b), target_names=['Not Fraud', 'Fraud']))


# final model performance tests

### ROC Curve and AUROC Score
#auroc score (area under the curve, closest to 1 is best)
pred = bst.predict(x_test)
pred_b = bst_b.predict(x_test_b)
auc = roc_auc_score(y_test, pred)
auc_b = roc_auc_score(y_test_b, pred_b)
print("AUROC score of imbalanced model: ",auc)
print("AUROC score of balanced model: ",auc_b)
print("\n")

#roc curve
fpr, tpr, _ = roc_curve(y_test, pred)
fpr_b, tpr_b, _ = roc_curve(y_test_b, pred_b)
plt.plot(fpr, tpr, marker='.', label="Imbalanced model (AUROC = %0.3f)" % auc)
plt.plot(fpr_b, tpr_b, marker='.', label="Balanced model (AUROC = %0.3f)" % auc_b)

plt.title("ROC Plot")
plt.xlabel("False Positive Rate")
plt.xlabel("True Positive Rate")
plt.legend()
plt.show()

### MCC Curve
from sklearn.metrics import matthews_corrcoef
print("Matthews Correlation Coefficient: Imbalanced Model = ", matthews_corrcoef(y_test, pred)) # scale of -1 to 1, 1 being perfect, 0 random, -1 inverse
print("Matthews Correlation Coefficient: Balanced Model = ", matthews_corrcoef(y_test_b, pred_b))
print("\n")

### Accuracy and Confusion Matrix
accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)
matrix = confusion_matrix(y_test, pred)

accuracy_b = accuracy_score(y_test_b, pred_b)
precision_b = precision_score(y_test_b, pred_b)
recall_b = recall_score(y_test_b, pred_b)
f1_b = f1_score(y_test_b, pred_b)
matrix_b = confusion_matrix(y_test_b, pred_b)

print("IMBALANCED REPORT - test data\n",classification_report(y_test, bst.predict(x_test), target_names=['Not Fraud', 'Fraud'], digits=4))
# print(f'Accuracy - Imbalanced: {accuracy*100:.2f}%')
print(f'Precision, Recall, F1-score - Imbalanced: {precision:.4f}, {recall:.4f}, {f1:.4f}')
print(f'Confusion Matrix - Imbalanced:')
print(matrix)
print("\n")

print("BALANCED REPORT - test data\n",classification_report(y_test_b, bst_b.predict(x_test_b), target_names=['Not Fraud', 'Fraud'], digits=4))
# print(f'Accuracy - Balanced: {accuracy_b*100:.2f}%')
print(f'Precision, Recall, F1-score - Balanced: {precision_b:.4f}, {recall_b:.4f}, {f1_b:.4f}')

print(f'Confusion Matrix - Balanced:')
print(matrix_b)


# calculating feature importance
importance_scores = bst.feature_importances_

# Print the scores along with feature names
for feature, score in zip(data.columns.values, importance_scores):
    print(f"{feature}: {score}") # most significant features: v14 > v10 > v12

