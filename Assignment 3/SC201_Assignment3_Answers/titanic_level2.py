"""
File: titanic_level2.py
Name: Jay
----------------------------------
This file builds a machine learning algorithm by pandas and sklearn libraries.
We'll be using pandas to read in dataset, store data into a DataFrame,
standardize the data by sklearn, and finally train the model and
test it on kaggle website. Hyper-parameters tuning are not required due to its
high level of abstraction, which makes it easier to use but less flexible.
You should find a good model that surpasses 77% test accuracy on kaggle.
"""

import math
import pandas as pd
from sklearn import preprocessing, linear_model

TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'

# Global variable
# Cache for test set missing data
nan_cache = {}


def data_preprocess(filename, mode='Train', training_data=None):
	"""
	:param filename: str, the filename to be read into pandas
	:param mode: str, indicating the mode we are using (either Train or Test)
	:param training_data: DataFrame, a 2D data structure that looks like an excel worksheet
						  (You will only use this when mode == 'Test')
	:return: Tuple(data, labels), if the mode is 'Train'; or return data, if the mode is 'Test'
	"""
	data = pd.read_csv(filename)
	labels = None

	# drop unnecessary columns
	del data["PassengerId"]
	del data["Name"]
	del data["Ticket"]
	del data["Cabin"]

	# replace categorical features with numerical values
	data['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)
	data['Sex'].replace(['female', 'male'], [0, 1], inplace=True)

	if mode == 'Train':

		# drop rows with missing data and select ture labels
		data.dropna(inplace=True)
		labels = data.pop('Survived')

		# calculate mean values (train data) for future use
		mean_age = round(data.Age.mean(), 3)
		mean_fare = round(data.Fare.mean(), 3)

		# Cache mean values for future use
		nan_cache["Age"] = mean_age
		nan_cache["Fare"] = mean_fare

		return data, labels
	elif mode == 'Test':
		# Fill missing data with mean values
		data.Age.fillna(nan_cache['Age'], inplace=True)
		data.Fare.fillna(nan_cache['Fare'], inplace=True)

		return data


def one_hot_encoding(data, feature):
	"""
	:param data: DataFrame, key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: DataFrame, remove the feature column and add its one-hot encoding features
	"""

	# One-hot encoding for categorical features
	if feature == 'Sex':
		data["Sex_0"] = [1 if val == 0 else 0 for val in data[feature]]
		data["Sex_1"] = [1 if val == 1 else 0 for val in data[feature]]
		data.pop("Sex")
	elif feature == 'Embarked':
		data["Embarked_0"] = [1 if val == 0 else 0 for val in data[feature]]
		data["Embarked_1"] = [1 if val == 1 else 0 for val in data[feature]]
		data["Embarked_2"] = [1 if val == 2 else 0 for val in data[feature]]
		data.pop("Embarked")
	elif feature == 'Pclass':
		data["Pclass_0"] = [1 if val == 1 else 0 for val in data[feature]]
		data["Pclass_1"] = [1 if val == 2 else 0 for val in data[feature]]
		data["Pclass_2"] = [1 if val == 3 else 0 for val in data[feature]]
		data.pop("Pclass")

	return data


def standardization(data, mode='Train'):
	"""
	:param data: DataFrame, key is the column name, value is its data
	:param mode: str, indicating the mode we are using (either Train or Test)
	:return data: DataFrame, standardized features
	"""

	# Standardize the features
	standard_scaler = preprocessing.StandardScaler()
	if mode == 'Train':
		standard_scaler.fit(data)
		data = standard_scaler.transform(data)
	elif mode == 'Test':
		data = standard_scaler.transform(data)
	return data


def main():
	"""
	You should call data_preprocess(), one_hot_encoding(), and
	standardization() on your training data. You should see ~80% accuracy on degree1;
	~83% on degree2; ~87% on degree3.
	Please write down the accuracy for degree1, 2, and 3 respectively below
	(rounding accuracies to 8 decimal places)
	TODO: real accuracy on degree1 -> 0.80196629
	TODO: real accuracy on degree2 -> 0.83707865
	TODO: real accuracy on degree3 -> 0.87840449
	"""
	# Load the dataset
	train_data, Y = data_preprocess(TRAIN_FILE, 'Train')

	# One-hot encoding
	one_hot_train_data = one_hot_encoding(train_data, 'Sex')
	one_hot_train_data = one_hot_encoding(one_hot_train_data, 'Embarked')
	one_hot_train_data = one_hot_encoding(one_hot_train_data, 'Pclass')

	# Standardization
	standard_scaler = preprocessing.StandardScaler()
	standard_scaler.fit(one_hot_train_data)
	standard_train_data = standard_scaler.transform(one_hot_train_data)

	# Polynomial features
	poly_phi = preprocessing.PolynomialFeatures(degree=2)
	train_data, Y = data_preprocess(TRAIN_FILE, 'Train')

	# One-hot encoding
	one_hot_train_data = one_hot_encoding(train_data, 'Sex')
	one_hot_train_data = one_hot_encoding(one_hot_train_data, 'Embarked')
	one_hot_train_data = one_hot_encoding(one_hot_train_data, 'Pclass')

	# Standardization
	standard_scaler = preprocessing.StandardScaler()
	standard_scaler.fit(one_hot_train_data)
	standard_train_data = standard_scaler.transform(one_hot_train_data)

	# Polynomial features
	poly_phi = preprocessing.PolynomialFeatures(degree=3)
	poly_train_data = poly_phi.fit_transform(standard_train_data)

	# Logistic Regression
	h = linear_model.LogisticRegression(max_iter=10000)

	# Train the model and evaluate it
	classifier = h.fit(poly_train_data, Y)
	accuracy = round(classifier.score(poly_train_data, Y), 8)
	print(f'Real accuracy: {accuracy}')
	return accuracy


if __name__ == '__main__':
	main()
