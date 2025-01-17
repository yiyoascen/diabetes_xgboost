# iris multiclass classification using LabelEncoder
from pandas import read_csv
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# load data
data = read_csv('iris.data', header=None)
dataset = data.values

# split data into X and y
# X gets all rows and all cols xept the label
X = dataset[:, 0:4]
# y gets all the labels
y = dataset[:, 4]

# encode string class values as integers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y)
label_encoded_y = label_encoder.transform(y)

seed = 7
test_size = .33
X_train, X_test, y_train, y_test = train_test_split(
    X, label_encoded_y, test_size=test_size, random_state=seed)

# fir model on training data
model = XGBClassifier()
model.fit(X_train, y_train)
print(model)

# make predictions for test data
predictions = model.predict(X_test)

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100))
