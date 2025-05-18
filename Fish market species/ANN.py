import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ds = pd.read_csv("Data/Original data/Fish_market.csv")
label_encoder = preprocessing.LabelEncoder()
ds['Species'] = label_encoder.fit_transform(ds['Species'])

def splitdataset(ds):
    X = ds.values[:, 1:7]
    Y = ds.values[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)
    return X, Y, X_train, X_test, y_train, y_test

X, Y, X_train, X_test, y_train, y_test = splitdataset(ds)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
    print(classification_report(y_test, y_pred, zero_division=0))

cal_accuracy(y_test, y_pred)

pd.DataFrame(y_pred, columns=["Predicted"]).to_csv("Prediction_ANN_model.csv", index=False)

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
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
