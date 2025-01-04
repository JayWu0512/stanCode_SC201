"""
File: titanic_level1.py
Name: Jay
----------------------------------
This file builds a machine learning algorithm from scratch 
by Python. We'll be using 'with open' to read in dataset,
store data into a Python dict, and finally train the model and 
test it on kaggle website. This model is the most flexible among all
levels. You should do hyper-parameter tuning to find the best model.
"""

import math
from util import *
from collections import defaultdict
TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'


def data_preprocess(filename: str, data: dict, mode='Train', training_data=None):
	"""
	:param filename: str, the filename to be processed
	:param data: an empty Python dictionary
	:param mode: str, indicating if it is training mode or testing mode
	:param training_data: dict[str: list], key is the column name, value is its data
						  (You will only use this when mode == 'Test')
	:return data: dict[str: list], key is the column name, value is its data
	"""
	is_header = True
	data = defaultdict(list)

	with open(filename, 'r') as f:
		for line in f:
			if mode == 'Train':
				if is_header:
					# origin = [PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked]
					# new = [Survived, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]
					header = line.strip().split(',')
					pop_list = [0, 3, 8, 10]
					for index in sorted(pop_list, reverse=True):
						header.pop(index)
					is_header = False
				else:
					train_data = line.strip().split(',')
					pop_list = [0, 3, 4, 9, 11]
					for index in sorted(pop_list, reverse=True):
						train_data.pop(index)

					# Append data to the dictionary if all elements are not empty
					if '' not in train_data:
						for i in range(len(header)):
							# Survived
							if i == 0:
								data[header[i]].append(int(train_data[i]))
							# Pclass
							if i == 1:
								data[header[i]].append(int(train_data[i]))
							# Sex
							if i == 2:
								if train_data[i] == 'male':
									data[header[i]].append(1)
								elif train_data[i] == 'female':
									data[header[i]].append(0)
							# Age
							if i == 3:
								data[header[i]].append(float(train_data[i]))
							# SibSp
							if i == 4:
								data[header[i]].append(int(train_data[i]))
							# Parch
							if i == 5:
								data[header[i]].append(int(train_data[i]))
							# Fare
							if i == 6:
								data[header[i]].append(float(train_data[i]))
							# Embarked
							if i == 7:
								if train_data[i] == 'S':
									data[header[i]].append(0)
								elif train_data[i] == 'C':
									data[header[i]].append(1)
								elif train_data[i] == 'Q':
									data[header[i]].append(2)

			else:
				if is_header:
					# origin = [PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked]
					# new = [Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]
					header = line.strip().split(',')
					pop_list = [0, 2, 7, 9]
					for index in sorted(pop_list, reverse=True):
						header.pop(index)
					is_header = False
				else:
					test_data = line.strip().split(',')
					pop_list = [0, 2, 3, 8, 10]
					for index in sorted(pop_list, reverse=True):
						test_data.pop(index)

					# Test data, so fill missing data with average of training data
					for i in range(len(header)):
						# Pclass
						if i == 0:
							if test_data[i] != "":
								data[header[i]].append(int(test_data[i]))
							# append average of training data
							else:
								data[header[i]].append(round(sum(training_data[header[i]])/len(training_data[header[i]]), 3))
						# Sex
						if i == 1:
							if test_data[i] == 'male':
								data[header[i]].append(1)
							elif test_data[i] == 'female':
								data[header[i]].append(0)
							# append mode of training data
							else:
								data[header[i]].append(0)
						# Age
						if i == 2:
							if test_data[i] != "":
								data[header[i]].append(float(test_data[i]))
							# append average of training data
							else:
								data[header[i]].append(round(sum(training_data[header[i]])/len(training_data[header[i]]), 3))
						# SibSp
						if i == 3:
							if test_data[i] != "":
								data[header[i]].append(int(test_data[i]))
							# append average of training data
							else:
								data[header[i]].append(round(sum(training_data[header[i]])/len(training_data[header[i]]), 3))
						# Parch
						if i == 4:
							if test_data[i] != "":
								data[header[i]].append(int(test_data[i]))
							# append average of training data
							else:
								data[header[i]].append(round(sum(training_data[header[i]])/len(training_data[header[i]]), 3))
						# Fare
						if i == 5:
							if test_data[i] != "":
								data[header[i]].append(float(test_data[i]))
							# append average of training data
							else:
								data[header[i]].append(
									round(sum(training_data[header[i]]) / len(training_data[header[i]]), 3))
						# Embarked
						if i == 6:
							if test_data[i] == 'S':
								data[header[i]].append(0)
							elif test_data[i] == 'C':
								data[header[i]].append(1)
							elif test_data[i] == 'Q':
								data[header[i]].append(2)
							# append mode of training data
							else:
								data[header[i]].append(0)
	return data


def one_hot_encoding(data: dict, feature: str):
	"""
	:param data: dict[str, list], key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: dict[str, list], remove the feature column and add its one-hot encoding features
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


def normalize(data: dict):
	"""
	:param data: dict[str, list], key is the column name, value is its data
	:return data: dict[str, list], key is the column name, value is its normalized data
	"""
	# Normalize the features that type in
	for feature in data:
		max_val = max(data[feature])
		min_val = min(data[feature])
		data[feature] = [(val - min_val) / (max_val - min_val) for val in data[feature]]

	return data


def learnPredictor(inputs: dict, labels: list, degree: int, num_epochs: int, alpha: float):
	"""
	:param inputs: dict[str, list], key is the column name, value is its data
	:param labels: list[int], indicating the true label for each data
	:param degree: int, degree of polynomial features
	:param num_epochs: int, the number of epochs for training
	:param alpha: float, known as step size or learning rate
	:return weights: dict[str, float], feature name and its weight
	"""
	# Step 1 : Initialize weights
	weights = {}  # feature => weight
	keys = list(inputs.keys())
	if degree == 1:
		for i in range(len(keys)):
			weights[keys[i]] = 0
	elif degree == 2:
		for i in range(len(keys)):
			weights[keys[i]] = 0
		for i in range(len(keys)):
			for j in range(i, len(keys)):
				weights[keys[i] + keys[j]] = 0

	# Step 2 : Create generated features to match the weights
	def generate_features(inputs: dict, degree: int, keys: list, index: int) -> dict:
		"""
		:param inputs: dict[str, list], key is the column name, value is its data
		:param degree: int, degree of polynomial features
		:param keys: list, name of features
		:param index: int, index of features
		:return: dict, generate feature vector
		"""
		features = {}
		# Degree 1
		for key in keys:
			features[key] = inputs[key][index]

		# Degree 2
		if degree == 2:
			for i in range(len(keys)):
				for j in range(i, len(keys)):
					feature_name = keys[i] + keys[j]
					features[feature_name] = inputs[keys[i]][index] * inputs[keys[j]][index]

		return features

	# Step 3 : Start to train
	num_samples = len(inputs[keys[0]])
	for epoch in range(num_epochs):
		for i in range(num_samples):
			current_features = generate_features(inputs, degree, keys, i)

			# Calculate h (classification problem)
			k = dotProduct(weights, current_features)
			h = 1/(1 + math.exp(-k))

			# true label
			y = labels[i]

			# S.G.D
			gradient = {key: current_features[key] for key in current_features}
			scale = -alpha * (h - y)
			increment(weights, scale, gradient)

	return weights
