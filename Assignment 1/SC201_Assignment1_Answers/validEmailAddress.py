"""
File: validEmailAddress.py
Name: Jay Wu
----------------------------
This file shows what a feature vector is
and what a weight vector is for valid email 
address classifier. You will use a given 
weight vector to classify what is the percentage
of correct classification.

Accuracy of this model: 0.6538461538461539
"""

WEIGHT = [                           # The weight vector selected by Jerry
	[0.4],                           # (see assignment handout for more details)
	[0.4],
	[0.2],
	[0.2],
	[0.9],
	[-0.65],
	[0.1],
	[0.1],
	[0.1],
	[-0.7]
]

DATA_FILE = 'is_valid_email.txt'     # This is the file name to be processed


def main():
	count = 0
	scores = []

	maybe_email_list = read_in_data()

	# calculate the feature vector in each email
	for maybe_email in maybe_email_list:
		feature_vector = feature_extractor(maybe_email)

		# calculate scores in each email
		score = sum(a[0] * b for a, b in zip(WEIGHT, feature_vector))
		scores.append(score)

	# calculate the correct rate
	# first 13 is the wrong email, if the score is <0, then count+1
	# last 13 is the correct email, if the score is >0, then count+1
	for i in range(len(scores)):
		if i <= 12 and scores[i] < 0:
			count += 1
		elif i > 12 and scores[i] > 0:
			count += 1

	# calculate the correct rate
	rate = count/len(scores)
	print(rate)


def feature_extractor(maybe_email):
	"""
	:param maybe_email: str, the string to be processed
	:return: list, feature vector with 10 values of 0's or 1's
	"""
	feature_vector = [0] * len(WEIGHT)
	for i in range(len(feature_vector)):

		# check if '@' in the str, if true = 1, if false = 0
		if i == 0:
			feature_vector[i] = 1 if '@' in maybe_email else 0

		# check no '.' before '@', if true = 1, if false = 0
		elif i == 1:
			if feature_vector[0]:
				feature_vector[i] = 1 if '.' not in maybe_email.split('@')[0] else 0

		# check some strings before '@', if true = 1, if false = 0
		elif i == 2:
			if feature_vector[0]:
				feature_vector[i] = 1 if maybe_email.split('@')[0] else 0

		# check some strings after '@', if true = 1, if false = 0
		elif i == 3:
			if feature_vector[0]:
				feature_vector[i] = 1 if maybe_email.split('@')[1].strip() != '' else 0

		# check there is '.' after '@', if true = 1, if false = 0
		elif i == 4:
			if feature_vector[0]:
				parts = maybe_email.split('@')[1:]
				feature_vector[i] = 1 if any('.' in part for part in parts) else 0

		# check there is no white space, if true = 1, if false = 0
		elif i == 5:
			feature_vector[i] = 1 if ' ' not in maybe_email else 0

		# check ends with '.com', if true = 1, if false = 0
		elif i == 6:
			feature_vector[i] = 1 if '.com' in maybe_email[-4:] else 0

		# check ends with '.edu', if true = 1, if false = 0
		elif i == 7:
			feature_vector[i] = 1 if '.edu' in maybe_email[-4:] else 0

		# check ends with '.tw', if true = 1, if false = 0
		elif i == 8:
			feature_vector[i] = 1 if '.tw' in maybe_email[-3:] else 0

		# check if length > 10, if true = 1, if false = 0
		elif i == 9:
			feature_vector[i] = 1 if len(maybe_email) > 10 else 0

	return feature_vector


def read_in_data():
	"""
	:return: list, containing strings that might be valid email addresses
	"""
	with open(DATA_FILE, 'r') as f:
		email = list(line.strip() for line in f.readlines())
	return email


if __name__ == '__main__':
	main()
