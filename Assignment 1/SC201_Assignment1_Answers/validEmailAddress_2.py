"""
File: validEmailAddress_2.py
Name: Jay
----------------------------
Please construct your own feature vectors
and try to surpass the accuracy achieved by
Jerry's feature vector in validEmailAddress.py.
feature1:  TODO: '@' in the str
feature2:  TODO: some strings before '@'
feature3:  TODO: some strings after '@'
feature4:  TODO: '.' cannot be in the first letter
feature5:  TODO: '.' cannot be in front of '@'
feature6:  TODO: no uppercase in the str
feature7:  TODO: '..' cannot be after '@'
feature8:  TODO: before '@', it must have English words or numbers
feature9:  TODO: in '"__"', it must have special symbols
feature10: TODO: adjust the score

Accuracy of your model: TODO: 1.0
"""

import numpy as np

WEIGHT = np.array([
    [1], [1], [1], [1], [1], [2], [3], [4], [5], [-17]
])

DATA_FILE = 'is_valid_email.txt'  # This is the file name to be processed


def main():
    count = 0
    scores = []

    maybe_email_list = read_in_data()
    for maybe_email in maybe_email_list:
        feature_vector = feature_extractor(maybe_email)

        # Transform the matrix(weight) to a horizontal orientation
        score = WEIGHT.T.dot(feature_vector)[0, 0]
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
    :return: list, feature vector with value 0's and 1's
    """
    # make a matric using feature vector
    feature_vector = np.zeros((10, 1))

    for i in range(len(feature_vector)):

        # check if '@' in the str
        if i == 0:
            feature_vector[i] = 1 if '@' in maybe_email else 0

        # check some strings before '@'
        elif i == 1:
            if feature_vector[0]:
                feature_vector[i] = 1 if maybe_email.split('@')[0] else 0

        # check some strings after '@'
        elif i == 2:
            if all(feature_vector[j] for j in range(1)):
                feature_vector[i] = 1 if maybe_email.split('@')[1].strip() != '' else 0

        # '.' cannot be in the first letter
        elif i == 3:
            if all(feature_vector[j] for j in range(2)):
                feature_vector[i] = 1 if not maybe_email.startswith('.') else 0

        # '.' cannot be in front of '@'
        elif i == 4:
            if all(feature_vector[j] for j in range(3)):
                feature_vector[i] = 1 if not maybe_email.split('@')[0].endswith('.') else 0

        # no uppercase in the str
        elif i == 5:
            if all(feature_vector[j] for j in range(4)):
                feature_vector[i] = 1 if maybe_email.islower() else 0

        # '..' cannot be after '@'
        elif i == 6:
            if all(feature_vector[j] for j in range(5)):
                parts = maybe_email.split('@')[1:]
                feature_vector[i] = 1 if not any('..' in part for part in parts) else 0

        # before '@', it must have English words or numbers
        elif i == 7:
            if all(feature_vector[j] for j in range(6)):
                feature_vector[i] = 1 if any(char.isalnum() for char in maybe_email.split('@')[0]) else 0

        # in '"__"', it must have special symbols
        elif i == 8:
            if all(feature_vector[j] for j in range(7)):
                if maybe_email.count('"') % 2 == 0 and maybe_email.count('"') >= 2:
                    quoted_part = maybe_email.split('"')[1]
                    feature_vector[i] = 1 if any(not char.isalnum() for char in quoted_part) else 0
                elif maybe_email.count('"') % 2 == 1:
                    feature_vector[i] = 0
                else:
                    feature_vector[i] = 1

        # adjust the score
        elif i == 9:
            feature_vector[i] = 1 if len(maybe_email) >= 0 else 0

    return feature_vector


def read_in_data():
    """
    :return: list, containing strings that may be valid email addresses
    """
    with open(DATA_FILE, 'r') as f:
        email = list(line.strip() for line in f.readlines())
    return email


if __name__ == '__main__':
    main()
