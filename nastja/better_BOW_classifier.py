# !/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer,LabelEncoder

def load_data(text):
    # open file with json string
    x = open(text,'r', encoding='utf-8')
    data = []
    labels = []
    # each line is an event
    for i,line in enumerate(x):
        # convert each event into dictionary
        instance = json.loads(line)
        # INSTANCE:
        # dictionary with keys: 'relation', 'entity_1', 'entity_2', 'snippet'
        ## RELATION:
        # relation labels: NO_REL, worked_at, author, has_spouse, capital
        # for each instance save label in list
        labels.append(instance['relation'])

        # INSTANCE['SNIPPET']:
        # list of 1 or more dictionaries
        instance_context = []
        for dictionary in instance['snippet']:
        ## DICTIONARY:
        # dictionary with 6 keys: left, mention_1, middle, mention_2, right, direction
            #### if I want to separate the context into three parts ####
            # left = [i.lower() for i in dictionary['left'].split()]
            # instance_context.append(left)
            # middle = [i.lower() for i in dictionary['middle'].split()]
            # instance_context.append(middle)
            # right = [i.lower() for i in dictionary['right'].split()]
            # instance_context.append(right)
            context_string = dictionary['left']+' '+dictionary['middle']+' '+dictionary['right']
            # lowercase and tokenize when necessary
            # instance_context = [i.lower() for i in context_string.split()]
            # print(type(context_string))

        data.append(context_string)

    return labels, data

def BOW_model(data):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data)
    return X

def LogReg(X, labels, X_test):
    logisticRegr = LogisticRegression()
    logisticRegr.fit(X, labels)
    predicted_labels = logisticRegr.predict(X_test[:10])

    return predicted_labels

def main():
    train = 'train.json.txt'
    test = 'test-covered.json.txt'
    labels, data = load_data(train)
    xxx, test_data = load_data(test)
    print(test_data[0])
    print(len(data), len(test_data))
    big_data = data+test_data
    print(len(big_data))
    X = BOW_model(big_data)
    print(X.shape)
    X_train = X[:9660]
    X_test = X[9660:]

    le = LabelEncoder()
    train_labels = le.fit_transform(labels)

    test_label_predicted = LogReg(X_train, train_labels, X_test)
    test_label_predicted_decoded = le.inverse_transform(test_label_predicted)
    print(test_label_predicted_decoded)

main()
