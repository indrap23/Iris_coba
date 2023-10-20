import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Read original dataset
iris_df = pd.read_csv('data/iris.csv')
iris_df.sample(frac=1, random_state=123)
# selecting features and target data
X = iris_df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
y = iris_df[['variety']]
# split data into train and test sets
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.3, random_state=123, stratify=y)
# create an instance of the random forest classifier
clf = RandomForestClassifier(n_estimators=100)
# train the classifier on the training data
clf.fit(X_train, y_train)
# predict on the test set
y_pred = clf.predict(X_test)
# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}') # Accuracy: 0.91


# save the model to disk
joblib.dump(clf, 'rf_model.sav')