"""
File: interactive.py
Name: Jay
------------------------
This file uses the function interactivePrompt
from util.py to predict the reviews input by 
users on Console. Remember to read the weights
and build a Dict[str: float]
"""

import util
import submission

TRAINED_WEIGHTS = 'weights'

def main():
	# open the file that had been trained previously
	with open(TRAINED_WEIGHTS, 'r') as f:
		weights = {line.split()[0]: float(line.split()[1]) for line in f}

	# apply the method to extract word at interactivePrompt
	feature_extractor = submission.extractWordFeatures

	# predict the words that users type is positive or negative ( 1 or -1) and its details
	util.interactivePrompt(feature_extractor, weights)


if __name__ == '__main__':
	main()
