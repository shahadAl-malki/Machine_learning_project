import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

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

def train_using_entropy(X_train, X_test, y_train):
    clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, min_samples_leaf=5)
    clf_entropy.fit(X_train, y_train)
    return clf_entropy

def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred

def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
    print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
    print(classification_report(y_test, y_pred, zero_division=0))

def plot_decision_tree(clf_object, feature_names, class_names):
    plt.figure(figsize=(15, 10))
    plot_tree(clf_object, filled=True, feature_names=feature_names, class_names=class_names, rounded=True)
    plt.show()

if __name__ == "__main__":
    data = ds_scaled
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)

    pd.DataFrame(X_train).to_csv("X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv("X_test.csv", index=False)
    pd.DataFrame(y_train).to_csv("Y_train.csv", index=False, header=["Species"])
    pd.DataFrame(y_test).to_csv("Y_test.csv", index=False, header=["Species"])

    clf_entropy = train_using_entropy(X_train, X_test, y_train)
    y_pred_entropy = prediction(X_test, clf_entropy)
    pd.DataFrame(y_pred_entropy, columns=["Predicted"]).to_csv("Prediction_Decision_tree.csv", index=False)
    cal_accuracy(y_test, y_pred_entropy)

    feature_names = ds_scaled.columns[1:7]
    class_names = ['Bream', 'Roach', 'Whitefish', 'Parkki', 'Perch', 'Pike', 'Smelt']
    plot_decision_tree(clf_entropy, feature_names, class_names)

    sns.heatmap(confusion_matrix(y_test, y_pred_entropy), annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    report = classification_report(y_test, y_pred_entropy, output_dict=True, zero_division=0)
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
