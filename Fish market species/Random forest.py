import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

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

X, Y, X_train, X_test, y_train, y_test = splitdataset(ds)

classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=42)
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
print(pred)

def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
    print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
    print(classification_report(y_test, y_pred, zero_division=0))

cal_accuracy(y_test, pred)

pd.DataFrame(pred, columns=["Predicted"]).to_csv("Prediction_Random_forest.csv", index=False)

sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

report = classification_report(y_test, pred, output_dict=True, zero_division=0)
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
