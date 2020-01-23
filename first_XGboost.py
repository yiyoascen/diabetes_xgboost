# my first use of XGboost with a diabetes dataset

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load data
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')

# split data into X and y

# X gets every value from every row except the last one
X = dataset[:, :8]
# y gets only the last vale from every row
y = dataset[:, 8]

# split data into train and test sets

# the seed is to get always the same random split
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                    random_state=seed)

# fit model on training data
model = XGBClassifier()
model.fit(X_train, y_train)

# you can see teh parameters used in a trained model by printing it
print(model)

# make predictions for test data
predictions = model.predict(X_test)

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print('accuracy: %.2f%%' % (accuracy * 100))
