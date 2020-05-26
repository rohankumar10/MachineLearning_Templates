# Artificial Neural Network
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Part 1 - Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
ct = ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder = 'passthrough')
X = ct.fit_transform(X)
X = X[:, 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 Lets Create ANN
# Import Keras library and other models of keras
# import keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Dropout
# Initializing the ANN
classifier = Sequential()

# Adding input layer and hidden layer
# uniform distribution so that weights are small numbers b/w 0 and 1
# units=6 As (Input_layer + Output_layer)/2 (Average)
# Activation function is rectifier func keyword is relu
# input_dim = No. of inputs so that the hidden layer we're creating knows what inputs to expect
classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu", input_dim=11))
classifier.add(Dropout(p = 0.1))
# We don't need to add input_dim for other hidden layers as after first hidden layer the 2nd will know what to expect
# Come on its Deep learning that is why another hidden layer
# Adding the 2nd Hidden layer
classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu"))
classifier.add(Dropout(p = 0.1))
# Adding the Output layer
# For DV having 3 categories we will need to choose softmax as activation and units as per categories
classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))

# Compile the whole ANN
# Optimizer is algorithm we want to use to find optimal set of weights
# Loss corresponds to loss func to find optimal weights sum of square of diff
# DV is binary then loss=binary_crossentropy
# DV has more than one category then loss=categorical_crossentropy
# Metrics criterion used to evaluate the model
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Choose the number of epoch
# Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)

# Part 3 Make the prediction and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Homework 
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Part 4 Evaulating, Improving and tuning ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers.core import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu", input_dim=11))
    classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu"))
    classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))
    classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return classifier    

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()

# Improving the ANN
# Dropout Regularization to reduce overfitting if needed
# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 150],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
