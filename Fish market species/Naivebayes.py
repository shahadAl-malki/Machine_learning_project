import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

ds = pd.read_csv("Data/Original data/Fish_market.csv")
label_encoder = preprocessing.LabelEncoder()
ds['Species'] = label_encoder.fit_transform(ds['Species'])
ds['Species'].unique()

def splitdataset(ds):
    X = ds.values[:, 1:7]
    Y = ds.values[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)
    return X, Y, X_train, X_test, y_train, y_test

X, Y, X_train, X_test, y_train, y_test = splitdataset(ds)

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)

def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
    print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
    print(classification_report(y_test, y_pred, zero_division=0))

cal_accuracy(y_test, y_pred)

pd.DataFrame(y_pred, columns=["Predicted"]).to_csv("Prediction_Naivebayes.csv", index=False)

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
