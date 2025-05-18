import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import svm

ds = pd.read_csv("Data/Original data/Fish_market.csv")
label_encoder = preprocessing.LabelEncoder()
ds['Species'] = label_encoder.fit_transform(ds['Species'])
ds['Species'].unique()
scaler = StandardScaler()
scaler.fit(ds[ds.columns[1:7]])
scaled_features = scaler.transform(ds[ds.columns[1:7]])
ds_scaled = pd.concat([ds[['Species']], pd.DataFrame(scaled_features, columns=ds.columns[1:7])], axis=1)

def splitdataset(ds):
    X = ds_scaled.values[:, 1:7]
    Y = ds_scaled.values[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)
    return X, Y, X_train, X_test, y_train, y_test

X, Y, X_train, X_test, y_train, y_test = splitdataset(ds_scaled)

clf = svm.SVC(kernel='rbf', C=1.0)
clf.fit(X_train, y_train)
y_pred_svm = clf.predict(X_test)
print(y_pred_svm)

def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
    print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
    print(classification_report(y_test, y_pred, zero_division=0))

cal_accuracy(y_test, y_pred_svm)

pd.DataFrame(y_pred_svm, columns=["Predicted"]).to_csv("Prediction_SVM_model.csv", index=False)

sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

report = classification_report(y_test, y_pred_svm, output_dict=True, zero_division=0)
df_report = pd.DataFrame(report).transpose()
df_report.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore', inplace=True)

df_report[['precision', 'recall', 'f1-score']].plot(kind='bar')
plt.title("Classification Report Metrics per Class")
plt.xlabel("Class Label")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
